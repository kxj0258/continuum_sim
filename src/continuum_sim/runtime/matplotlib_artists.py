"""Reusable Matplotlib artist pools for low-frequency presentation sinks."""

from __future__ import annotations


class PersistentAxisArtists:
    """Provide an Axes-like API that reuses lines and reference artists."""

    def __init__(self, axis) -> None:
        self.axis = axis
        self._lines = []
        self._references = []
        self._dynamic = []
        self._line_index = 0
        self._reference_index = 0
        self._legend = None

    def begin_frame(self) -> None:
        self._line_index = 0
        self._reference_index = 0
        for artist in self._dynamic:
            try:
                artist.remove()
            except (AttributeError, ValueError):
                pass
        self._dynamic = []

    def end_frame(self) -> None:
        for artist in self._lines[self._line_index :]:
            artist.set_visible(False)
        for artist in self._references[self._reference_index :]:
            artist.set_visible(False)

    def plot(self, *args, **kwargs):
        index = self._line_index
        self._line_index += 1
        if index >= len(self._lines):
            artist = self.axis.plot(*args, **kwargs)[0]
            self._lines.append(artist)
            return [artist]
        artist = self._lines[index]
        artist.set_visible(True)
        if hasattr(artist, "set_data_3d") and len(args) >= 3 and not isinstance(
            args[2], str
        ):
            artist.set_data_3d(args[0], args[1], args[2])
        else:
            artist.set_data(args[0], args[1])
        if kwargs:
            artist.set(**kwargs)
        return [artist]

    def semilogy(self, *args, **kwargs):
        self.axis.set_yscale("log")
        return self.plot(*args, **kwargs)

    def axhline(self, y=0.0, **kwargs):
        index = self._reference_index
        self._reference_index += 1
        if index >= len(self._references):
            artist = self.axis.axhline(y, **kwargs)
            self._references.append(artist)
            return artist
        artist = self._references[index]
        artist.set_visible(True)
        artist.set_ydata([y, y])
        if kwargs:
            artist.set(**kwargs)
        return artist

    def scatter(self, *args, **kwargs):
        artist = self.axis.scatter(*args, **kwargs)
        self._dynamic.append(artist)
        return artist

    def axvline(self, *args, **kwargs):
        artist = self.axis.axvline(*args, **kwargs)
        self._dynamic.append(artist)
        return artist

    def axvspan(self, *args, **kwargs):
        artist = self.axis.axvspan(*args, **kwargs)
        self._dynamic.append(artist)
        return artist

    def text(self, *args, **kwargs):
        artist = self.axis.text(*args, **kwargs)
        self._dynamic.append(artist)
        return artist

    def legend(self, *args, **kwargs):
        if self._legend is None:
            self._legend = self.axis.legend(*args, **kwargs)
        return self._legend

    def __getattr__(self, name: str):
        return getattr(self.axis, name)


__all__ = ["PersistentAxisArtists"]
