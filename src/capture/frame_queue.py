"""Fila concorrente de capacidade 1 com política drop-oldest."""

import threading


class LatestQueue[T]:
    """Guarda só o item mais recente: `put` nunca bloqueia e descarta o anterior não lido.

    O consumidor sempre recebe o dado mais novo, então atraso não se acumula
    quando ele fica mais lento que o produtor.
    """

    def __init__(self) -> None:
        self._item: T | None = None
        self._has_item = False
        self._cond = threading.Condition()
        self.dropped = 0

    def put(self, item: T) -> None:
        with self._cond:
            if self._has_item:
                self.dropped += 1
            self._item, self._has_item = item, True
            self._cond.notify()

    def get(self, timeout: float | None = None) -> T | None:
        """Espera um item novo; devolve `None` se o tempo acabar."""
        with self._cond:
            if not self._cond.wait_for(lambda: self._has_item, timeout):
                return None
            item, self._item, self._has_item = self._item, None, False
            return item
