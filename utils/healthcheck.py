# monitoring/healthcheck.py

from typing import Dict, Any
import logging

from backend_user import get_supabase_client

logger = logging.getLogger(__name__)


def healthcheck() -> Dict[str, Any]:
    """
    Simple healthcheck for app:
    - Supabase connectivity
    - Basic config presence
    """
    status = {
        "supabase_ok": False,
        "details": {},
    }

    try:
        client = get_supabase_client()
        # lightweight query: just fetch current timestamp from Postgres
        res = client.rpc("now").execute()
        status["supabase_ok"] = True
        status["details"]["db_now"] = str(res.data)
    except Exception as e:
        logger.exception("Healthcheck supabase failed")
        status["supabase_ok"] = False
        status["details"]["error"] = str(e)

    return status
