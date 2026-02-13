"""YouTube Transcript Fetcher using yt-dlp."""

import json
import re
import subprocess
from pathlib import Path
from typing import Optional

import yt_dlp

from ..config import get_settings


class TranscriptFetcher:
    """Fetches transcripts from YouTube videos using yt-dlp."""

    def __init__(self) -> None:
        """Initialize the transcript fetcher."""
        self.settings = get_settings()
        self.transcript_dir = self.settings.get_transcript_path()

    def _sanitize_filename(self, title: str) -> str:
        """Sanitize a string to be used as a filename."""
        sanitized = re.sub(r'[<>:"/\\|?*]', "_", title)
        sanitized = sanitized[:100]
        return sanitized

    def _extract_video_id(self, url: str) -> Optional[str]:
        """Extract video ID from various YouTube URL formats."""
        patterns = [
            r"(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([a-zA-Z0-9_-]{11})",
            r"youtube\.com/shorts/([a-zA-Z0-9_-]{11})",
        ]
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None

    def _get_subtitles(
        self, ydl: yt_dlp.YoutubeDL, video_url: str
    ) -> tuple[Optional[dict], Optional[str]]:
        """Extract subtitles from a video."""
        try:
            info = ydl.extract_info(video_url, download=False)
            if info is None:
                return None, None

            subtitles = info.get("subtitles")
            automatic_captions = info.get("automatic_captions")

            if subtitles:
                return subtitles, info.get("title", "Unknown")
            elif automatic_captions:
                return automatic_captions, info.get("title", "Unknown")

            return None, info.get("title", "Unknown")
        except Exception:
            return None, None

    def _download_subtitle_file(
        self, ydl: yt_dlp.YoutubeDL, video_url: str, subtitle_lang: str
    ) -> Optional[str]:
        """Download subtitle file directly."""
        ydl_opts = {
            "writesubtitles": True,
            "writeautomaticsub": True,
            "subtitleslangs": [subtitle_lang],
            "skip_download": True,
            "quiet": True,
            "no_warnings": True,
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(video_url, download=False)
                if info:
                    return info.get("title", "Unknown")
        except Exception:
            pass
        return None

    def fetch_transcript(self, url: str, output_path: Optional[Path] = None) -> str:
        """
        Fetch transcript from a YouTube video or playlist.

        Args:
            url: YouTube video or playlist URL
            output_path: Optional path to save the transcript

        Returns:
            Path to the saved transcript file

        Raises:
            ValueError: If no transcripts are available
            RuntimeError: If yt-dlp fails
        """
        ydl_opts = {
            "skip_download": True,
            "quiet": True,
            "no_warnings": True,
            "extract_flat": False,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)

            if info is None:
                raise ValueError(f"Could not extract info from: {url}")

            if "entries" in info:
                return self._fetch_playlist(url, output_path)
            else:
                return self._fetch_single_video(url, info, output_path)

    def _fetch_playlist(self, playlist_url: str, output_path: Optional[Path] = None) -> str:
        """Fetch transcripts from all videos in a playlist."""
        ydl_opts = {
            "skip_download": True,
            "quiet": True,
            "no_warnings": True,
            "extract_flat": False,
        }

        all_transcripts: list[str] = []
        playlist_title = "Playlist"

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(playlist_url, download=False)
            if info is None:
                raise ValueError(f"Could not extract playlist: {playlist_url}")

            playlist_title = info.get("title", "Playlist")
            entries = info.get("entries", [])

            for i, entry in enumerate(entries):
                if entry is None:
                    continue
                video_url = f"https://www.youtube.com/watch?v={entry['id']}"
                try:
                    transcript = self._extract_transcript_text(ydl, video_url)
                    if transcript:
                        all_transcripts.append(
                            f"--- Video {i + 1}: {entry.get('title', 'Unknown')} ---\n{transcript}"
                        )
                except Exception:
                    continue

        if not all_transcripts:
            raise ValueError(f"No transcripts found in playlist: {playlist_title}")

        combined_text = "\n\n".join(all_transcripts)

        if output_path is None:
            sanitized = self._sanitize_filename(playlist_title)
            output_path = self.transcript_dir / f"{sanitized}.txt"

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(combined_text, encoding="utf-8")

        return str(output_path)

    def _fetch_single_video(
        self,
        video_url: str,
        info: dict,
        output_path: Optional[Path] = None,
    ) -> str:
        """Fetch transcript from a single video."""
        transcript = self._extract_transcript_text(
            yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True}), video_url
        )

        if not transcript:
            raise ValueError(
                f"No transcripts available for: {video_url}. "
                "The video may not have subtitles or captions."
            )

        if output_path is None:
            title = info.get("title", "Unknown")
            sanitized = self._sanitize_filename(title)
            output_path = self.transcript_dir / f"{sanitized}.txt"

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(transcript, encoding="utf-8")

        return str(output_path)

    def _extract_transcript_text(self, ydl: yt_dlp.YoutubeDL, video_url: str) -> Optional[str]:
        """Extract transcript text from a video."""
        try:
            info = ydl.extract_info(video_url, download=False)
            if info is None:
                return None

            subtitles = info.get("subtitles") or info.get("automatic_captions")

            if not subtitles:
                return None

            available_langs = list(subtitles.keys())

            preferred_langs = ["en", "en-US", "en-GB", "en-CA", "en-AU"]
            selected_lang = None
            for lang in preferred_langs:
                if lang in available_langs:
                    selected_lang = lang
                    break

            if not selected_lang and available_langs:
                selected_lang = available_langs[0]

            if not selected_lang:
                return None

            subtitle_data = subtitles[selected_lang]

            if isinstance(subtitle_data, list) and subtitle_data:
                subtitle_data = subtitle_data[0]

            if isinstance(subtitle_data, dict):
                ext = subtitle_data.get("ext", "srt")
                url = subtitle_data.get("url")

                if url:
                    try:
                        import requests

                        response = requests.get(url, timeout=10)
                        if response.status_code == 200:
                            if ext == "json3":
                                return self._parse_json3_subtitles(response.text)
                            elif ext == "srt":
                                return self._parse_srt_subtitles(response.text)
                    except Exception:
                        pass

            return None

        except Exception:
            return None

    def _parse_json3_subtitles(self, json3_text: str) -> str:
        """Parse JSON3 format subtitles."""
        try:
            data = json.loads(json3_text)
            lines: list[str] = []

            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        segs = item.get("segs", [])
                        for seg in segs:
                            if isinstance(seg, dict):
                                text = seg.get("text", "")
                                if text:
                                    lines.append(text.strip())

            return " ".join(lines)
        except Exception:
            return ""

    def _parse_srt_subtitles(self, srt_text: str) -> str:
        """Parse SRT format subtitles."""
        lines: list[str] = []
        blocks = srt_text.strip().split("\n\n")

        for block in blocks:
            block_lines = block.strip().split("\n")
            for line in block_lines[2 if len(block_lines) > 2 else 0 :]:
                if line.strip() and not line.strip().isdigit():
                    lines.append(line.strip())

        return " ".join(lines)

    def get_video_info(self, url: str) -> dict:
        """
        Get video information without fetching transcript.

        Args:
            url: YouTube video URL

        Returns:
            Dictionary with video information
        """
        ydl_opts = {
            "skip_download": True,
            "quiet": True,
            "no_warnings": True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            if info is None:
                raise ValueError(f"Could not extract info from: {url}")

            return {
                "id": info.get("id"),
                "title": info.get("title"),
                "description": info.get("description", "")[:500],
                "duration": info.get("duration"),
                "uploader": info.get("uploader"),
                "upload_date": info.get("upload_date"),
                "view_count": info.get("view_count"),
                "like_count": info.get("like_count"),
                "has_subtitles": bool(info.get("subtitles") or info.get("automatic_captions")),
            }
