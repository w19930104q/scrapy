from __future__ import annotations

import asyncio
import functools
import logging
import warnings
from abc import ABC, abstractmethod
from collections import OrderedDict, defaultdict
from traceback import clear_frames
from typing import TYPE_CHECKING, Any, Literal, Self, TypeAlias, TypedDict, cast

from twisted.internet.defer import CancelledError, Deferred, DeferredList
from twisted.python.failure import Failure

from scrapy.exceptions import ScrapyDeprecationWarning
from scrapy.http.request import NO_CALLBACK, Request
from scrapy.utils._datatypes import SequenceExclude
from scrapy.utils._deps_compat import TWISTED_FAILURE_HAS_STACK
from scrapy.utils.asyncio import is_asyncio_available
from scrapy.utils.decorators import _warn_spider_arg
from scrapy.utils.defer import (
    _process_pending_io,
    deferred_from_coro,
    ensure_awaitable,
    maybe_deferred_to_future,
)
from scrapy.utils.log import failure_to_exc_info
from scrapy.utils.misc import arg_to_iter
from scrapy.utils.python import global_object_name

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from scrapy import Spider
    from scrapy.crawler import Crawler
    from scrapy.http import Response
    from scrapy.settings import Settings
    from scrapy.utils.request import RequestFingerprinterProtocol


class FileInfo(TypedDict):
    url: str
    path: str
    checksum: str | None
    status: str


FileInfoOrError: TypeAlias = (
    tuple[Literal[True], FileInfo] | tuple[Literal[False], Failure]
)

logger = logging.getLogger(__name__)


class FileException(Exception):
    """General media error exception"""


class _MediaRequestFiltered(FileException):
    """Raised internally by media pipelines when a media request is filtered
    out (e.g. as an offsite request) instead of being downloaded.

    It is a subclass of :exc:`FileException` for backward compatibility, but
    unlike an actual download error it is logged at the ``DEBUG`` level and
    without a traceback, since filtering a request is expected behavior rather
    than an error.
    """


def _media_request_filtered(failure: Failure) -> bool:
    return isinstance(failure.value, _MediaRequestFiltered)


class _Waiter(TypedDict):
    deferred: Deferred[FileInfo]
    errback: Callable[[Failure], FileInfo] | None
    active: bool


class _DownloadState:
    """A media download that is in progress and may be shared by several items.

    Every item whose request has the same fingerprint waits on its own
    :class:`~twisted.internet.defer.Deferred` in *waiting*. The first such item
    runs the download through *deferred*; cancelling an individual item only
    deactivates its own waiters, while cancelling the last active waiter
    cancels *deferred* and thus the shared download itself.
    """

    def __init__(
        self,
        deferred: Deferred[FileInfo | Failure],
        fp: bytes,
        info: MediaPipeline.SpiderInfo,
    ) -> None:
        self.deferred = deferred
        self.fp = fp
        self.info = info
        self.waiting: list[_Waiter] = []

    @property
    def active_waiters(self) -> list[_Waiter]:
        return [waiter for waiter in self.waiting if waiter["active"]]


class MediaPipeline(ABC):
    LOG_FAILED_RESULTS: bool = True

    class SpiderInfo:
        def __init__(self, spider: Spider):
            self.spider: Spider = spider
            self.downloading: set[bytes] = set()
            self.downloaded: OrderedDict[bytes, FileInfo | Failure] = OrderedDict()
            # One entry per fingerprint with a download in progress. The
            # ``downloading`` set keeps the same fingerprints for backwards
            # compatibility of the public state inspection.
            self.download_states: dict[bytes, _DownloadState] = {}

        @property
        def waiting(self) -> defaultdict[bytes, list[Deferred[FileInfo]]]:
            # Backwards-compatible view over the per-download waiter lists:
            # only Deferreds of items still waiting for a result, grouped by
            # fingerprint, in registration order.
            grouped: defaultdict[bytes, list[Deferred[FileInfo]]] = defaultdict(list)
            for fp, state in self.download_states.items():
                grouped[fp] = [
                    waiter["deferred"] for waiter in state.waiting if waiter["active"]
                ]
            return grouped

    def __init__(
        self,
        download_func: None = None,
        *,
        crawler: Crawler,
    ):
        if download_func is not None:  # pragma: no cover
            warnings.warn(
                "The download_func argument of MediaPipeline.__init__() is ignored"
                " and will be removed in a future Scrapy version.",
                category=ScrapyDeprecationWarning,
                stacklevel=2,
            )
        self.crawler: Crawler = crawler
        self._fingerprinter: RequestFingerprinterProtocol = (
            crawler.request_fingerprinter
        )

        settings = crawler.settings
        resolve = functools.partial(
            self._key_for_pipe, base_class_name="MediaPipeline", settings=settings
        )
        self._redirect_http_codes: set[int] = {
            int(x) for x in settings.getlist("REDIRECT_HTTP_CODES")
        }
        self.allow_redirects: bool = settings.getbool(
            resolve("MEDIA_ALLOW_REDIRECTS"), False
        )
        self._handle_statuses(self.allow_redirects)
        self._cache_size: int = settings.getint(resolve("MEDIA_CACHE_SIZE"))

    def _handle_statuses(self, allow_redirects: bool) -> None:
        self.handle_httpstatus_list = None
        if allow_redirects:
            # Statuses that RedirectMiddleware may redirect, i.e. those from
            # REDIRECT_HTTP_CODES, are left for it to handle.
            self.handle_httpstatus_list = SequenceExclude(
                {*range(300, 400), *self._redirect_http_codes}
            )

    def _key_for_pipe(
        self,
        key: str,
        base_class_name: str | None = None,
        settings: Settings | None = None,
    ) -> str:
        class_name = self.__class__.__name__
        formatted_key = f"{class_name.upper()}_{key}"
        if (
            not base_class_name
            or class_name == base_class_name
            or (settings and not settings.get(formatted_key))
        ):
            return key
        return formatted_key

    @classmethod
    def from_crawler(cls, crawler: Crawler) -> Self:
        return cls(crawler=crawler)

    @_warn_spider_arg
    def open_spider(self, spider: Spider | None = None) -> None:
        assert self.crawler.spider
        self.spiderinfo = self.SpiderInfo(self.crawler.spider)

    @_warn_spider_arg
    async def process_item(self, item: Any, spider: Spider | None = None) -> Any:
        info = self.spiderinfo
        requests = arg_to_iter(self.get_media_requests(item, info))
        coros = [self._process_request(r, info, item) for r in requests]
        results: list[FileInfoOrError] = []
        if coros:
            if is_asyncio_available():
                results_asyncio = await asyncio.gather(*coros, return_exceptions=True)
                for res in results_asyncio:
                    if isinstance(res, BaseException):
                        results.append((False, Failure(res)))
                    else:
                        results.append((True, res))
            else:
                deferreds = [deferred_from_coro(coro) for coro in coros]

                def _consume_cancellation(
                    failure: Failure,
                ) -> Failure:
                    # DeferredList still receives this failure, but consuming
                    # it here keeps the cancelled Deferred from reporting an
                    # unhandled error when process_item() re-raises it.
                    failure.trap(CancelledError)
                    return failure

                for deferred in deferreds:
                    deferred.addErrback(_consume_cancellation)
                results = await cast(
                    "Deferred[list[FileInfoOrError]]",
                    DeferredList(deferreds, consumeErrors=True),
                )
                # DeferredList turns a cancellation into an ordinary result,
                # but a cancelled item must end with CancelledError before
                # item_completed() rather than being processed.
                for _, value in results:
                    if isinstance(value, Failure) and value.check(CancelledError):
                        value.raiseException()
        return await ensure_awaitable(self.item_completed(results, item, info))

    async def _process_request(
        self, request: Request, info: SpiderInfo, item: Any
    ) -> FileInfo:
        fp = self._fingerprinter.fingerprint(request)

        eb: Callable[[Failure], FileInfo] | None = request.errback
        request.callback = NO_CALLBACK
        request.errback = None

        # Return cached result if request was already seen
        if fp in info.downloaded:
            await _process_pending_io()
            info.downloaded.move_to_end(fp)
            cached_result = info.downloaded[fp]
            if isinstance(cached_result, Failure):
                if eb:
                    return eb(cached_result)
                cached_result.raiseException()
            return cached_result

        # Otherwise, wait for result. Each item owns its own Deferred, so
        # cancelling one item only deactivates its own waiters without
        # touching the items sharing the download. The request errback is not
        # attached to the Deferred: a cancelled item must fail with
        # CancelledError instead of letting the errback recover it.
        wad: Deferred[FileInfo] = Deferred(lambda d: self._on_waiter_canceled(d, info))
        waiter: _Waiter = {"deferred": wad, "errback": eb, "active": True}

        state = info.download_states.get(fp)
        if state is None:
            # Start the download, checking the media_to_download hook first.
            info.downloading.add(fp)
            state = _DownloadState(
                deferred_from_coro(self._download_media(request, info, item)),
                fp,
                info,
            )
            info.download_states[fp] = state
            # A late download result is ignored unless it comes from the
            # download still registered for this fingerprint.
            state.deferred.addBoth(
                lambda result: self._download_finished(result, state, info)
            )
        state.waiting.append(waiter)
        return await maybe_deferred_to_future(wad)  # it must return wad at last

    async def _download_media(
        self, request: Request, info: SpiderInfo, item: Any
    ) -> FileInfo | Failure:
        try:
            file_info: FileInfo | None = await ensure_awaitable(
                self.media_to_download(request, info, item=item)
            )
            if file_info:
                # got a result without downloading
                return file_info
            # download the result
            return await self._check_media_to_download(request, info, item=item)
        except (CancelledError, asyncio.CancelledError):
            # A cancelled download is not a media failure: propagate it so the
            # item ends with CancelledError instead of reaching
            # media_failed() or item_completed().
            raise
        except Exception:
            failure = Failure()
            if not _media_request_filtered(failure):
                logger.exception(failure)
            return failure

    def _download_finished(
        self, result: FileInfo | Failure, state: _DownloadState, info: SpiderInfo
    ) -> None:
        # The download may have been cancelled with its last waiter and a new
        # download for the same fingerprint may have started since; only the
        # download still registered owns the state. A cancelled download is
        # always detached beforehand, so its late result is ignored here.
        if info.download_states.get(state.fp) is not state:
            return
        info.downloading.discard(state.fp)
        del info.download_states[state.fp]
        self._cache_result_and_execute_waiters(result, state)

    def _on_waiter_canceled(self, wad: Deferred[FileInfo], info: SpiderInfo) -> None:
        # Cancelling an item deactivates only its own waiters; one item can
        # hold several waiters for the same fingerprint (duplicate URLs), and
        # cancelling its task reaches all of them in a chain. The shared
        # download is cancelled only once no active waiter remains, leaving the
        # other items untouched. item_completed() is never called for a
        # cancelled item.
        for state in list(info.download_states.values()):
            for waiter in state.waiting:
                if waiter["deferred"] is not wad or not waiter["active"]:
                    continue
                waiter["active"] = False
                if (
                    not state.active_waiters
                    and info.download_states.get(state.fp) is state
                ):
                    info.downloading.discard(state.fp)
                    del info.download_states[state.fp]
                    state.deferred.cancel()
                return

    def _modify_media_request(self, request: Request) -> None:
        if self.handle_httpstatus_list:
            request.meta["handle_httpstatus_list"] = self.handle_httpstatus_list
        else:
            request.meta["handle_httpstatus_all"] = True

    async def _check_media_to_download(
        self, request: Request, info: SpiderInfo, item: Any
    ) -> FileInfo:
        try:
            self._modify_media_request(request)
            response = await self.crawler.engine.download_async(request)
            return await ensure_awaitable(
                self.media_downloaded(response, request, info, item=item)
            )
        except Exception:
            failure = self.media_failed(Failure(), request, info)
            if isinstance(failure, Failure):
                warnings.warn(
                    f"{global_object_name(self.media_failed)} returned a Failure instance."
                    f" This is deprecated, please raise an exception instead, e.g. via failure.raiseException().",
                    category=ScrapyDeprecationWarning,
                    stacklevel=2,
                )
                failure.raiseException()

    def _cache_result_and_execute_waiters(
        self, result: FileInfo | Failure, state: _DownloadState
    ) -> None:
        if isinstance(result, Failure):
            # minimize cached information for failure
            exception: BaseException | None = result.value
            while exception is not None:
                clear_frames(exception.__traceback__)
                exception = exception.__context__
            result.cleanFailure()
            result.frames.clear()
            if TWISTED_FAILURE_HAS_STACK:
                result.stack.clear()

        info = state.info
        info.downloaded[state.fp] = result  # cache result
        if 0 <= self._cache_size < len(info.downloaded):
            info.downloaded.popitem(last=False)
        for waiter in state.waiting:
            if not waiter["active"]:
                # Its item was cancelled while the download was in flight.
                continue
            wad = waiter["deferred"]
            if isinstance(result, Failure):
                errback = waiter["errback"]
                if errback:
                    # Let the request errback handle the failure; if it
                    # returns a value, the item gets a successful result.
                    try:
                        wad.callback(errback(result))
                    except Exception:
                        wad.errback(Failure())
                else:
                    wad.errback(result)
            else:
                wad.callback(result)

    # Overridable Interface
    @abstractmethod
    def media_to_download(
        self, request: Request, info: SpiderInfo, *, item: Any = None
    ) -> Deferred[FileInfo | None] | None:
        """Check request before starting download"""
        raise NotImplementedError

    @abstractmethod
    def get_media_requests(self, item: Any, info: SpiderInfo) -> list[Request]:
        """Returns the media requests to download"""
        raise NotImplementedError

    @abstractmethod
    def media_downloaded(
        self,
        response: Response,
        request: Request,
        info: SpiderInfo,
        *,
        item: Any = None,
    ) -> FileInfo | Awaitable[FileInfo]:
        """Handler for success downloads"""
        raise NotImplementedError

    @abstractmethod
    def media_failed(
        self, failure: Failure, request: Request, info: SpiderInfo
    ) -> Failure:
        """Handler for failed downloads"""
        raise NotImplementedError

    def item_completed(
        self, results: list[FileInfoOrError], item: Any, info: SpiderInfo
    ) -> Any:
        """Called per item when all media requests has been processed"""
        if self.LOG_FAILED_RESULTS:
            for ok, value in results:
                if not ok:
                    assert isinstance(value, Failure)
                    if _media_request_filtered(value):
                        continue
                    logger.error(
                        "%(class)s found errors processing %(item)s",
                        {"class": self.__class__.__name__, "item": item},
                        exc_info=failure_to_exc_info(value),
                        extra={"spider": info.spider},
                    )
        return item

    @abstractmethod
    def file_path(
        self,
        request: Request,
        response: Response | None = None,
        info: SpiderInfo | None = None,
        *,
        item: Any = None,
    ) -> str:
        """Returns the path where downloaded media should be stored"""
        raise NotImplementedError
