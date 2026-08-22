class GenerationProgress:
    """Prints a named generation stage, its progress, and its final outcome."""

    _active_progress = None

    def __init__(self, generator, stage, iteration=0):
        self.generator = generator
        self.stage = stage
        self.iteration = iteration
        self._last_line_length = 0

    @property
    def prefix(self):
        return f"[{self.generator} {self.iteration}]"

    def start(self, details=""):
        """Start a new generation stage, printing its name and optional details."""
        self._close_other_active_line()
        suffix = f" — {details}" if details else ""
        print(f"{self.prefix} {self.stage}: started{suffix}")

    def _close_other_active_line(self):
        """Close any other active progress line, if it exists, to avoid overwriting it."""
        active = GenerationProgress._active_progress
        if active is not None and active is not self and active._last_line_length:
            print()
            active._last_line_length = 0
        if active is not self:
            GenerationProgress._active_progress = None

    def update(self, completed, total=None, details=""):
        """Update the progress of the current generation stage, printing the completed and total counts."""
        self._close_other_active_line()
        if total is None:
            progress = f"{completed} completed"
        elif total == 0:
            progress = "0/0 (100.0%)"
        else:
            progress = f"{completed}/{total} ({completed / total * 100:.1f}%)"
        suffix = f" | {details}" if details else ""
        line = f"{self.prefix} {self.stage}: {progress}{suffix}"
        padding = " " * max(0, self._last_line_length - len(line))
        print(f"\r{line}{padding}", end="", flush=True)
        self._last_line_length = len(line)
        GenerationProgress._active_progress = self

    def finish(self, success, details=""):
        """Finish the current generation stage, printing its final outcome and optional details."""
        self._close_other_active_line()
        outcome = "successful" if success else "failed"
        suffix = f" — {details}" if details else ""
        line = f"{self.prefix} {self.stage}: {outcome}{suffix}"
        padding = " " * max(0, self._last_line_length - len(line))
        print(f"\r{line}{padding}")
        self._last_line_length = 0
        if GenerationProgress._active_progress is self:
            GenerationProgress._active_progress = None

    def warning(self, details):
        """Print a warning without leaving an in-place progress line behind."""
        self._close_other_active_line()
        padding = " " * self._last_line_length
        print(f"\r{padding}\r{self.prefix} {self.stage}: warning — {details}")
        self._last_line_length = 0
        if GenerationProgress._active_progress is self:
            GenerationProgress._active_progress = None
