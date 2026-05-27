"""Chat channels module with plugin architecture."""

from finabot.channels.base import BaseChannel
from finabot.channels.manager import ChannelManager

__all__ = ["BaseChannel", "ChannelManager"]