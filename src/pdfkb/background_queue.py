"""Background processing queue for PDF tasks.

Provides asynchronous job handling with priority, concurrency limits,
status tracking, retry logic, and graceful shutdown.

Other components can import :class:`BackgroundProcessingQueue`,
:class:`JobType`, :class:`JobStatus`, and :class:`Priority`.
"""

import asyncio
import logging
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Awaitable, Callable, Dict, Optional

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class JobType(Enum):
    FILE_WATCHER = auto()
    WEB_UPLOAD = auto()


class JobStatus(Enum):
    QUEUED = auto()
    PROCESSING = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELED = auto()


class Priority(Enum):
    HIGH = 0
    NORMAL = 1
    LOW = 2


@dataclass
class Job:
    job_id: str
    job_type: JobType
    metadata: Dict[str, Any]
    priority: Priority
    status: JobStatus = JobStatus.QUEUED
    attempts: int = 0
    max_retries: int = 3
    # Optional callback that actually processes the job.
    # It must be an async callable accepting the Job instance.
    processor: Optional[Callable[["Job"], Awaitable[None]]] = None


class BackgroundProcessingQueue:
    """Async job queue with priority and worker management.

    Example
    -------
    >>> async def dummy_processor(job: Job):
    ...     await asyncio.sleep(0.1)  # simulate work
    ...
    >>> queue = BackgroundProcessingQueue(concurrency=2, default_processor=dummy_processor)
    >>> job_id = await queue.add_job(
    ...     job_type=JobType.WEB_UPLOAD,
    ...     metadata={"filename": "example.pdf"},
    ...     priority=Priority.HIGH,
    ... )
    >>> status = await queue.get_status(job_id)
    """

    def __init__(
        self,
        concurrency: int = 4,
        default_processor: Optional[Callable[[Job], Awaitable[None]]] = None,
        max_retries: int = 3,
        thread_pool_size: int = 2,
    ) -> None:
        self._queue: asyncio.PriorityQueue[Job] = asyncio.PriorityQueue()
        self._jobs: Dict[str, Job] = {}
        self._concurrency = concurrency
        self._workers: list[asyncio.Task] = []
        self._shutdown_event = asyncio.Event()
        self._default_processor = default_processor
        self._max_retries = max_retries
        self._stats: Dict[JobStatus, int] = defaultdict(int)
        self._counter = 0  # Counter to avoid Job comparison issues

        # Thread pool for CPU-intensive operations
        self._thread_pool = ThreadPoolExecutor(max_workers=thread_pool_size)
        self._use_thread_pool_for_types = {JobType.WEB_UPLOAD, JobType.FILE_WATCHER}

        for _ in range(self._concurrency):
            worker = asyncio.create_task(self._worker_loop())
            self._workers.append(worker)

    async def _worker_loop(self) -> None:
        while not self._shutdown_event.is_set():
            try:
                # Wait for a job; timeout allows checking shutdown flag.
                priority_job_tuple = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                _, _, job = priority_job_tuple  # Extract job from (priority, counter, job) tuple
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                # Mark any pending job as done when cancelled
                try:
                    self._queue.task_done()
                except ValueError:
                    pass
                break

            # Check shutdown flag again before processing
            if self._shutdown_event.is_set() or job.status == JobStatus.CANCELED:
                self._queue.task_done()
                continue

            job.status = JobStatus.PROCESSING
            self._update_stats()
            processor = job.processor or self._default_processor

            try:
                if processor is None:
                    raise RuntimeError("No processor defined for job %s" % job.job_id)

                # Check shutdown flag before starting processor
                if self._shutdown_event.is_set():
                    job.status = JobStatus.CANCELED
                    self._queue.task_done()
                    continue

                # Run CPU-intensive jobs (PDF processing) in thread pool
                if job.job_type in self._use_thread_pool_for_types:
                    logger.info(
                        f"🧵 BACKGROUND QUEUE: Running job {job.job_id} ({job.job_type.name}) "
                        f"in thread pool to avoid blocking web server"
                    )
                    await asyncio.get_event_loop().run_in_executor(
                        self._thread_pool, self._run_processor_sync, processor, job
                    )
                    logger.info(f"🧵 BACKGROUND QUEUE: Job {job.job_id} completed in thread pool")
                else:
                    # Run other jobs directly in event loop
                    logger.info(
                        f"⚡ BACKGROUND QUEUE: Running job {job.job_id} ({job.job_type.name}) in main event loop"
                    )
                    await processor(job)

                # Check if we should still mark as completed (might have been cancelled)
                if not self._shutdown_event.is_set():
                    job.status = JobStatus.COMPLETED
                    logger.info("Job %s completed", job.job_id)
                else:
                    job.status = JobStatus.CANCELED
                    logger.info("Job %s cancelled during processing", job.job_id)

            except asyncio.CancelledError:
                job.status = JobStatus.CANCELED
                logger.info("Job %s cancelled", job.job_id)
            except Exception as exc:  # noqa: BLE001
                job.attempts += 1
                logger.exception("Job %s failed on attempt %s: %s", job.job_id, job.attempts, exc)

                # Don't retry if shutting down
                if not self._shutdown_event.is_set() and job.attempts < (job.max_retries or self._max_retries):
                    job.status = JobStatus.QUEUED
                    self._counter += 1
                    await self._queue.put((job.priority.value, self._counter, job))
                    logger.info("Re‑queued job %s (attempt %s)", job.job_id, job.attempts)
                else:
                    job.status = JobStatus.FAILED
                    logger.error("Job %s marked as FAILED after %s attempts", job.job_id, job.attempts)

            self._update_stats()
            self._queue.task_done()

    def _update_stats(self) -> None:
        # Reset and recount
        self._stats = defaultdict(int)
        for job in self._jobs.values():
            self._stats[job.status] += 1

    async def add_job(
        self,
        job_type: JobType,
        metadata: Dict[str, Any],
        priority: Priority = Priority.NORMAL,
        processor: Optional[Callable[[Job], Awaitable[None]]] = None,
        max_retries: Optional[int] = None,
    ) -> str:
        """Enqueue a new job.

        Returns
        -------
        str
            Unique job identifier.
        """
        job_id = str(uuid.uuid4())
        job = Job(
            job_id=job_id,
            job_type=job_type,
            metadata=metadata,
            priority=priority,
            processor=processor,
            max_retries=max_retries or self._max_retries,
        )
        self._jobs[job_id] = job
        self._counter += 1
        await self._queue.put((priority.value, self._counter, job))
        self._update_stats()
        logger.info("Enqueued job %s with priority %s", job_id, priority.name)
        return job_id

    async def get_status(self, job_id: str) -> Optional[JobStatus]:
        """Return the current status of a job, or ``None`` if unknown."""
        job = self._jobs.get(job_id)
        return job.status if job else None

    async def cancel_job(self, job_id: str) -> bool:
        """Attempt to cancel a queued job.

        Returns ``True`` if the job was found and cancelled, ``False`` otherwise.
        """
        job = self._jobs.get(job_id)
        if not job:
            return False
        if job.status in {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELED}:
            return False
        job.status = JobStatus.CANCELED
        self._update_stats()
        logger.info("Cancelled job %s", job_id)
        return True

    async def get_statistics(self) -> Dict[JobStatus, int]:
        """Snapshot of job counts per status."""
        # Return a copy to avoid external mutation.
        return dict(self._stats)

    def _run_processor_sync(self, processor: Callable[[Job], Awaitable[None]], job: Job) -> None:
        """Run an async processor function synchronously in a thread.

        This method is called in the thread pool to execute CPU-intensive processors.
        """
        try:
            # Create a new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(processor(job))
            finally:
                loop.close()
        except Exception as e:
            logger.error(f"Error running processor in thread for job {job.job_id}: {e}")
            raise

    async def shutdown(self, wait: bool = True, timeout: float = 5.0) -> None:
        """Signal workers to stop and optionally wait for the queue to drain.

        Parameters
        ----------
        wait: bool, default True
            If ``True``, the method will wait for all queued jobs to be processed
            (or cancelled) before returning.
        timeout: float, default 5.0
            Maximum time to wait for jobs to complete before forcing shutdown.
        """
        logger.info("Initiating background queue shutdown...")
        self._shutdown_event.set()

        if wait:
            try:
                # Wait for queue to drain with timeout
                logger.info(f"Waiting for {self._queue.qsize()} queued jobs to complete (timeout: {timeout}s)...")
                await asyncio.wait_for(self._queue.join(), timeout=timeout)
                logger.info("All queued jobs completed")
            except asyncio.TimeoutError:
                logger.warning(
                    f"Timeout ({timeout}s) reached, forcing shutdown with {self._queue.qsize()} jobs remaining"
                )
                # Cancel any remaining jobs
                await self._cancel_remaining_jobs()

        # Cancel all worker tasks
        logger.info(f"Cancelling {len(self._workers)} worker tasks...")
        for w in self._workers:
            if not w.done():
                w.cancel()

        # Wait for workers to finish with a timeout
        try:
            await asyncio.wait_for(asyncio.gather(*self._workers, return_exceptions=True), timeout=2.0)
            logger.info("All worker tasks cancelled successfully")
        except asyncio.TimeoutError:
            logger.warning("Worker task cancellation timeout, proceeding with shutdown")

        # Shutdown thread pool with shorter wait time for responsiveness
        logger.info("Shutting down thread pool...")
        try:
            # ThreadPoolExecutor.shutdown() doesn't have a timeout parameter
            # So we run it with asyncio timeout wrapper
            await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(None, lambda: self._thread_pool.shutdown(wait=True)),
                timeout=3.0,
            )
            logger.info("Thread pool shut down successfully")
        except asyncio.TimeoutError:
            logger.warning("Thread pool shutdown timeout, forcing termination")
            # Force shutdown without waiting
            self._thread_pool.shutdown(wait=False)
        except Exception as e:
            logger.warning(f"Thread pool shutdown error (continuing): {e}")

        logger.info("BackgroundProcessingQueue shut down complete")

    async def _cancel_remaining_jobs(self) -> None:
        """Cancel all remaining jobs in the queue."""
        cancelled_count = 0

        # Cancel jobs that are still queued
        temp_jobs = []
        try:
            while True:
                try:
                    priority_job_tuple = self._queue.get_nowait()
                    _, _, job = priority_job_tuple
                    if job.status == JobStatus.QUEUED:
                        job.status = JobStatus.CANCELED
                        cancelled_count += 1
                    temp_jobs.append(priority_job_tuple)
                except asyncio.QueueEmpty:
                    break
        except Exception as e:
            logger.warning(f"Error during job cancellation: {e}")

        # Put non-cancelled jobs back (though we're shutting down)
        for job_tuple in temp_jobs:
            try:
                self._queue.task_done()  # Mark each extracted job as done
            except ValueError:
                pass  # Queue might be empty or task_done called too many times

        if cancelled_count > 0:
            logger.info(f"Cancelled {cancelled_count} remaining jobs")

        self._update_stats()


# Convenience singleton for the application (optional)
# from .config import settings  # Example import if you have a settings module
# default_queue = BackgroundProcessingQueue(concurrency=settings.max_workers)

__all__ = [
    "BackgroundProcessingQueue",
    "JobType",
    "JobStatus",
    "Priority",
    "Job",
]
