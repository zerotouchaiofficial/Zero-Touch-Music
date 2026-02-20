"""
check_copyright.py
Checks if an uploaded video is blocked or has copyright issues.
"""

import time
import logging
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

log = logging.getLogger("yt-uploader")


def check_video_status(youtube, video_id: str, max_wait: int = 60) -> dict:
    """
    Checks if video is blocked/restricted.
    
    Returns dict with:
    - blocked: bool (True if completely blocked)
    - restricted: bool (True if partially blocked in some countries)
    - status: str (description)
    """
    log.info(f"  Checking copyright status for video {video_id}...")
    
    # Wait a bit for YouTube to process
    time.sleep(10)
    
    try:
        response = youtube.videos().list(
            part="status,contentDetails",
            id=video_id
        ).execute()
        
        if not response.get("items"):
            return {"blocked": True, "restricted": False, "status": "Video not found"}
        
        item = response["items"][0]
        status = item.get("status", {})
        content_details = item.get("contentDetails", {})
        
        # Check upload status
        upload_status = status.get("uploadStatus", "")
        if upload_status == "rejected":
            return {"blocked": True, "restricted": False, "status": "Rejected by YouTube"}
        
        # Check if video is embeddable (blocked videos often aren't)
        embeddable = status.get("embeddable", True)
        
        # Check content rating/restrictions
        region_restriction = content_details.get("regionRestriction", {})
        blocked_regions = region_restriction.get("blocked", [])
        allowed_regions = region_restriction.get("allowed", [])
        
        # Check if blocked
        if upload_status == "failed":
            return {"blocked": True, "restricted": False, "status": "Upload failed"}
        
        # Check if restricted
        if blocked_regions or (allowed_regions and len(allowed_regions) < 50):
            return {
                "blocked": False, 
                "restricted": True, 
                "status": f"Blocked in {len(blocked_regions)} regions"
            }
        
        # All good
        return {"blocked": False, "restricted": False, "status": "Available worldwide"}
        
    except HttpError as e:
        log.warning(f"  Error checking status: {e}")
        return {"blocked": False, "restricted": False, "status": "Unknown (couldn't check)"}


def delete_video(youtube, video_id: str) -> bool:
    """Deletes a video. Returns True if successful."""
    try:
        youtube.videos().delete(id=video_id).execute()
        log.info(f"  ✓ Deleted video {video_id}")
        return True
    except HttpError as e:
        log.warning(f"  Failed to delete video: {e}")
        return False
