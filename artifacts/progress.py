class GenerationProgress:
    """Render roadmap-generation progress in one consistent terminal format."""

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
        print(f"{self.prefix} {self.stage}: STARTED{suffix}")

    def _close_other_active_line(self):
        """Close any other active progress line, if it exists, to avoid overwriting it."""
        active = GenerationProgress._active_progress
        if active is not None and active is not self and active._last_line_length:
            print()
            active._last_line_length = 0
        if active is not self:
            GenerationProgress._active_progress = None

    @staticmethod
    def _format_count(completed, total):
        if total is None:
            return f"{completed} completed"
        if total == 0:
            return "0/0 (100.0%)"
        return f"{completed}/{total} ({completed / total * 100:.1f}%)"

    def update(self, completed, total=None, details=""):
        """Redraw the current generation stage, including a percentage when total is known."""
        self._close_other_active_line()
        progress = self._format_count(completed, total)
        suffix = f" | {details}" if details else ""
        line = f"{self.prefix} {self.stage}: PROGRESS — {progress}{suffix}"
        padding = " " * max(0, self._last_line_length - len(line))
        print(f"\r{line}{padding}", end="", flush=True)
        self._last_line_length = len(line)
        GenerationProgress._active_progress = self

    def finish(self, success, details="", completed=None, total=None):
        """Finish a stage, retaining its final count and percentage when available."""
        self._close_other_active_line()
        outcome = "SUCCESS" if success else "FAILURE"
        progress = ""
        if completed is not None or total is not None:
            progress = f" — {self._format_count(completed or 0, total)}"
        suffix = f" | {details}" if details else ""
        line = f"{self.prefix} {self.stage}: {outcome}{progress}{suffix}"
        padding = " " * max(0, self._last_line_length - len(line))
        print(f"\r{line}{padding}")
        self._last_line_length = 0
        if GenerationProgress._active_progress is self:
            GenerationProgress._active_progress = None

    def warning(self, details):
        """Print a warning without leaving an in-place progress line behind."""
        self._close_other_active_line()
        padding = " " * self._last_line_length
        print(f"\r{padding}\r{self.prefix} {self.stage}: WARNING — {details}")
        self._last_line_length = 0
        if GenerationProgress._active_progress is self:
            GenerationProgress._active_progress = None


class ProgressReportingAlgorithm:
    """Provides consistently labelled, configurable progress reporting to algorithms."""

    progress_name = None

    def _create_progress(self, stage):
        """Create a progress reporter only when output was explicitly requested."""
        if not self.print_progress:
            return None
        return GenerationProgress(self.progress_name, stage, self.iteration)
