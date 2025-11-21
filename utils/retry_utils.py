import time
import logging

logger = logging.getLogger("utils.retry_utils")


def retry_with_backoff(
    fn,
    max_retries=3,
    exceptions=(Exception,),
    base_delay=0.1,
):
    """
    Retry a function with exponential backoff.

    Expected behavior (as tests require):
    - initial call + max_retries attempts
      e.g., max_retries=3 → total calls = 4
    """
    attempt = 0
    while True:
        try:
            return fn()
        except exceptions as e:
            if attempt >= max_retries:
                # After exhausting all retries → re-raise
                raise

            attempt += 1
            wait = base_delay * (2 ** (attempt - 1))
            logger.warning(f"Retry {attempt}/{max_retries} failed: {str(e)}")
            time.sleep(wait)
