"""Sobreposição da mistura GPU com o passo de treino (`--gpu-mix-overlap`).

Uma thread puxa a próxima receita do DataLoader (memória pinned), copia (H2D, `non_blocking`) e mistura o lote k+1
numa `torch.cuda.Stream` própria enquanto o passo k roda na stream padrão. Entrega `(lote, nº de amostras refeitas na
CPU)` já pronto: o laço de treino só espera o que faltou.

Sincronização (sem corrida):
  * a stream de consumo faz `wait_event` no evento gravado no fim da mistura (copia + kernels + fill_invalid);
  * cada tensor CUDA do lote recebe `record_stream(consumo)`: o alocador, que alocou na stream lateral, não reutiliza
    a memória enquanto o passo ainda a usa;
  * as receitas pinned só voltam ao pool depois que a stream lateral terminou de ler (o evento é sincronizado
    na thread antes de soltar a referência `rb`);
  * profundidade 1: no máximo um lote pronto e um sendo misturado (2 lotes misturados em memória, ~B*n*... KB);
  * exceções da thread são relançadas no `next`; `close()` encerra a thread.
"""

from __future__ import annotations

import queue
import threading

import torch


class MixPrefetcher:
    def __init__(self, it, mixer, ds, device: str | torch.device, depth: int = 1):
        self.q: queue.Queue = queue.Queue(maxsize=depth)
        self.dev = torch.device(device)
        self.stream = torch.cuda.Stream(self.dev)
        self.mixer, self.ds, self.it = mixer, ds, it
        self._stop = threading.Event()
        self.th = threading.Thread(target=self._run, name="mix-prefetch", daemon=True)
        self.th.start()

    def _put(self, item) -> bool:
        while not self._stop.is_set():
            try:
                self.q.put(item, timeout=0.2)
                return True
            except queue.Full:
                continue
        return False

    def _run(self):
        try:
            with torch.cuda.device(self.dev), torch.cuda.stream(self.stream):
                for rb in self.it:
                    b = self.mixer(rb)
                    nr = self.mixer.fill_invalid(b, self.ds)
                    b.pop("valid")
                    b.pop("index")
                    ev = torch.cuda.Event()
                    ev.record(self.stream)
                    ev.synchronize()  # a receita pinned `rb` só é solta depois da leitura (copia H2D) terminar
                    del rb
                    if not self._put((b, ev, nr)):
                        return
            self._put(None)
        except BaseException as e:  # noqa: BLE001
            self._put(e)

    def __iter__(self):
        return self

    def __next__(self):
        item = self.q.get()
        if item is None:
            raise StopIteration
        if isinstance(item, BaseException):
            raise item
        b, ev, nr = item
        cur = torch.cuda.current_stream(self.dev)
        cur.wait_event(ev)
        for v in b.values():
            if torch.is_tensor(v) and v.is_cuda:
                v.record_stream(cur)
        return b, nr

    def close(self):
        self._stop.set()
        self.th.join(timeout=5)
