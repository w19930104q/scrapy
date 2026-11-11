from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import MagicMock

import pytest
from twisted.internet.defer import CancelledError, Deferred, succeed
from twisted.python.failure import Failure

from scrapy import signals
from scrapy.exceptions import ScrapyDeprecationWarning
from scrapy.http import Request, Response
from scrapy.pipelines.media import (
    FileInfo,
    FileInfoOrError,
    MediaPipeline,
    _MediaRequestFiltered,
)
from scrapy.utils.asyncio import sleep
from scrapy.utils.defer import (
    _process_pending_io,
    deferred_from_coro,
    maybe_deferred_to_future,
)
from scrapy.utils.log import failure_to_exc_info
from scrapy.utils.misc import build_from_crawler
from scrapy.utils.signal import disconnect_all
from scrapy.utils.spider import DefaultSpider
from scrapy.utils.test import get_crawler
from tests.utils.decorators import coroutine_test
from tests.utils.media_pipelines import mocked_download_func

if TYPE_CHECKING:
    from collections.abc import Awaitable

    from scrapy.crawler import Crawler


class UserDefinedPipeline(MediaPipeline):
    def media_to_download(
        self, request: Request, info: MediaPipeline.SpiderInfo, *, item: Any = None
    ) -> Deferred[FileInfo | None] | None:
        return None

    def get_media_requests(
        self, item: Any, info: MediaPipeline.SpiderInfo
    ) -> list[Request]:
        return []

    def media_downloaded(
        self,
        response: Response,
        request: Request,
        info: MediaPipeline.SpiderInfo,
        *,
        item: Any = None,
    ) -> FileInfo | Awaitable[FileInfo]:
        return cast("FileInfo", {})

    def media_failed(
        self, failure: Failure, request: Request, info: MediaPipeline.SpiderInfo
    ) -> Failure:
        failure.raiseException()

    def file_path(
        self,
        request: Request,
        response: Response | None = None,
        info: MediaPipeline.SpiderInfo | None = None,
        *,
        item: Any = None,
    ) -> str:
        return ""


class TestBaseMediaPipeline:
    pipeline_class = UserDefinedPipeline
    settings = None

    def setup_method(self):
        crawler = get_crawler(DefaultSpider, self.settings)
        crawler.spider = crawler._create_spider()
        crawler.engine = MagicMock(download_async=mocked_download_func)
        self.pipe = build_from_crawler(self.pipeline_class, crawler)
        self.pipe.open_spider()
        self.info = self.pipe.spiderinfo
        self.fingerprint = crawler.request_fingerprinter.fingerprint

    @property
    def mocked_pipe(self) -> MockedMediaPipeline:
        assert isinstance(self.pipe, MockedMediaPipeline)
        return self.pipe

    def teardown_method(self):
        for name, signal in vars(signals).items():
            if not name.startswith("_"):
                disconnect_all(signal)

    def test_modify_media_request(self):
        request = Request("http://url")
        self.pipe._modify_media_request(request)
        assert request.meta == {"handle_httpstatus_all": True}

    def test_default_item_completed(self, caplog: pytest.LogCaptureFixture) -> None:
        item = {"name": "name"}
        assert self.pipe.item_completed([], item, self.info) is item

        # Check that failures are logged by default
        fail = Failure(Exception())
        results: Any = [(True, 1), (False, fail)]

        caplog.clear()
        new_item = self.pipe.item_completed(results, item, self.info)
        assert new_item is item
        assert len(caplog.records) == 1
        record = caplog.records[0]
        assert record.levelname == "ERROR"
        assert record.exc_info == failure_to_exc_info(fail)

        # disable failure logging and check again
        caplog.clear()
        self.pipe.LOG_FAILED_RESULTS = False
        new_item = self.pipe.item_completed(results, item, self.info)
        assert new_item is item
        assert len(caplog.records) == 0

    def test_item_completed_filtered_request_not_logged(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Filtered media requests (e.g. offsite ones) are not logged as errors
        by item_completed(), as they are not download errors."""
        item = {"name": "name"}
        fail = Failure(_MediaRequestFiltered("Filtered offsite request"))
        results: Any = [(True, 1), (False, fail)]

        with caplog.at_level(logging.DEBUG):
            new_item = self.pipe.item_completed(results, item, self.info)

        assert new_item is item
        assert len(caplog.records) == 0

    @coroutine_test
    async def test_default_process_item(self):
        item = {"name": "name"}
        new_item = await self.pipe.process_item(item)
        assert new_item is item


class DeferredItemCompletedPipeline(UserDefinedPipeline):
    def item_completed(
        self, results: list[FileInfoOrError], item: Any, info: MediaPipeline.SpiderInfo
    ) -> Deferred[Any]:
        return succeed(item)


class AsyncItemCompletedPipeline(UserDefinedPipeline):
    async def item_completed(
        self, results: list[FileInfoOrError], item: Any, info: MediaPipeline.SpiderInfo
    ) -> Any:
        return item


@pytest.mark.parametrize(
    "pipeline_class", [DeferredItemCompletedPipeline, AsyncItemCompletedPipeline]
)
@coroutine_test
async def test_awaitable_item_completed(pipeline_class: type[MediaPipeline]) -> None:
    crawler = get_crawler(DefaultSpider)
    crawler.spider = crawler._create_spider()
    pipe = build_from_crawler(pipeline_class, crawler)
    pipe.open_spider()
    item = {"name": "name"}
    assert await pipe.process_item(item) is item


class MockedMediaPipeline(UserDefinedPipeline):
    def __init__(self, *args: Any, crawler: Crawler, **kwargs: Any):
        super().__init__(*args, crawler=crawler, **kwargs)
        self._mockcalled: list[str] = []

    def media_to_download(
        self, request: Request, info: MediaPipeline.SpiderInfo, *, item: Any = None
    ) -> Deferred[FileInfo | None] | None:
        self._mockcalled.append("media_to_download")
        if "result" in request.meta:
            return request.meta.get("result")
        return super().media_to_download(request, info)

    def get_media_requests(
        self, item: Any, info: MediaPipeline.SpiderInfo
    ) -> list[Request]:
        self._mockcalled.append("get_media_requests")
        return item.get("requests")  # type: ignore[no-any-return]

    def media_downloaded(
        self,
        response: Response,
        request: Request,
        info: MediaPipeline.SpiderInfo,
        *,
        item: Any = None,
    ) -> FileInfo | Awaitable[FileInfo]:
        self._mockcalled.append("media_downloaded")
        return super().media_downloaded(response, request, info)

    def media_failed(
        self, failure: Failure, request: Request, info: MediaPipeline.SpiderInfo
    ) -> Failure:
        self._mockcalled.append("media_failed")
        return super().media_failed(failure, request, info)

    def item_completed(
        self, results: list[FileInfoOrError], item: Any, info: MediaPipeline.SpiderInfo
    ) -> Any:
        self._mockcalled.append("item_completed")
        item = super().item_completed(results, item, info)
        item["results"] = results
        return item


class AsyncMediaDownloadedPipeline(MockedMediaPipeline):
    async def media_downloaded(  # type: ignore[override]
        self,
        response: Response,
        request: Request,
        info: MediaPipeline.SpiderInfo,
        *,
        item: Any = None,
    ) -> FileInfo | Awaitable[FileInfo]:
        return super().media_downloaded(response, request, info)


class TestMediaPipeline(TestBaseMediaPipeline):
    pipeline_class = MockedMediaPipeline

    def _errback(self, result):
        self.mocked_pipe._mockcalled.append("request_errback")
        return result

    @coroutine_test
    async def test_result_succeed(self):
        rsp = Response("http://url1")
        req = Request(
            "http://url1",
            meta={"response": rsp},
            errback=self._errback,
        )
        item = {"requests": req}
        new_item = await self.pipe.process_item(item)
        assert new_item["results"] == [(True, {})]
        assert self.mocked_pipe._mockcalled == [
            "get_media_requests",
            "media_to_download",
            "media_downloaded",
            "item_completed",
        ]

    @coroutine_test
    async def test_result_failure(self):
        self.pipe.LOG_FAILED_RESULTS = False
        exc = Exception("foo")
        req = Request(
            "http://url1",
            meta={"response": exc},
            errback=self._errback,
        )
        item = {"requests": req}
        new_item = await self.pipe.process_item(item)
        assert len(new_item["results"]) == 1
        assert new_item["results"][0][0] is False
        assert isinstance(new_item["results"][0][1], Failure)
        assert new_item["results"][0][1].value == exc
        assert self.mocked_pipe._mockcalled == [
            "get_media_requests",
            "media_to_download",
            "media_failed",
            "request_errback",
            "item_completed",
        ]

    @coroutine_test
    async def test_mix_of_success_and_failure(self):
        self.pipe.LOG_FAILED_RESULTS = False
        rsp1 = Response("http://url1")
        req1 = Request("http://url1", meta={"response": rsp1})
        exc = Exception("foo")
        req2 = Request("http://url2", meta={"response": exc})
        item = {"requests": [req1, req2]}
        new_item = await self.pipe.process_item(item)
        assert len(new_item["results"]) == 2
        assert new_item["results"][0] == (True, {})
        assert new_item["results"][1][0] is False
        assert isinstance(new_item["results"][1][1], Failure)
        assert new_item["results"][1][1].value == exc
        m = self.mocked_pipe._mockcalled
        # only once
        assert m[0] == "get_media_requests"  # first hook called
        assert m.count("get_media_requests") == 1
        assert m.count("item_completed") == 1
        assert m[-1] == "item_completed"  # last hook called
        # twice, one per request
        assert m.count("media_to_download") == 2
        # one to handle success and other for failure
        assert m.count("media_downloaded") == 1
        assert m.count("media_failed") == 1

    @coroutine_test
    async def test_get_media_requests(self):
        # returns single Request (without callback)
        req = Request("http://url")
        item = {"requests": req}  # pass a single item
        new_item = await self.pipe.process_item(item)
        assert new_item is item
        assert self.fingerprint(req) in self.info.downloaded

        # returns iterable of Requests
        req1 = Request("http://url1")
        req2 = Request("http://url2")
        item = {"requests": iter([req1, req2])}  # type: ignore[dict-item]
        new_item = await self.pipe.process_item(item)
        assert new_item is item
        assert self.fingerprint(req1) in self.info.downloaded
        assert self.fingerprint(req2) in self.info.downloaded

    @coroutine_test
    async def test_results_are_cached_across_multiple_items(self):
        rsp1 = Response("http://url1")
        req1 = Request("http://url1", meta={"response": rsp1})
        item: dict[str, Any] = {"requests": req1}
        new_item = await self.pipe.process_item(item)
        assert new_item is item
        assert new_item["results"] == [(True, {})]

        # rsp2 is ignored, rsp1 must be in results because request fingerprints are the same
        req2 = Request(
            req1.url, meta={"response": Response("http://donot.download.me")}
        )
        item = {"requests": req2}
        new_item = await self.pipe.process_item(item)
        assert new_item is item
        assert self.fingerprint(req1) == self.fingerprint(req2)
        assert new_item["results"] == [(True, {})]

    @coroutine_test
    async def test_failures_are_cached_across_multiple_items(self):
        self.pipe.LOG_FAILED_RESULTS = False
        exc = Exception("foo")
        req1 = Request("http://url1", meta={"response": exc})
        new_item = await self.pipe.process_item({"requests": req1})
        assert new_item["results"][0][1].value is exc

        # rsp2 is ignored, the cached failure must be reused because request
        # fingerprints are the same
        req2 = Request(
            req1.url, meta={"response": Response("http://donot.download.me")}
        )
        new_item = await self.pipe.process_item({"requests": req2})
        assert new_item["results"][0][0] is False
        assert new_item["results"][0][1].value is exc
        assert self.mocked_pipe._mockcalled.count("media_to_download") == 1

    @coroutine_test
    async def test_cached_failure_calls_errback(self):
        """The errback of a request is called for a cached failure as well."""
        self.pipe.LOG_FAILED_RESULTS = False
        exc = Exception("foo")
        await self.pipe.process_item(
            {"requests": Request("http://url1", meta={"response": exc})}
        )

        def errback(failure):
            self.mocked_pipe._mockcalled.append("request_errback")
            return {"recovered": failure.value}

        req = Request("http://url1", errback=errback)
        new_item = await self.pipe.process_item({"requests": req})
        assert new_item["results"] == [(True, {"recovered": exc})]
        assert self.mocked_pipe._mockcalled.count("request_errback") == 1

    @coroutine_test
    async def test_results_are_cached_for_requests_of_single_item(self):
        rsp1 = Response("http://url1")
        req1 = Request("http://url1", meta={"response": rsp1})
        req2 = Request(
            req1.url, meta={"response": Response("http://donot.download.me")}
        )
        item: dict[str, Any] = {"requests": [req1, req2]}
        new_item = await self.pipe.process_item(item)
        assert new_item is item
        assert new_item["results"] == [(True, {}), (True, {})]

    @coroutine_test
    async def test_wait_if_request_is_downloading(self):
        def _check_downloading(response: Response) -> Response:
            fp = self.fingerprint(req1)
            assert fp in self.info.downloading
            assert fp in self.info.waiting
            assert fp not in self.info.downloaded
            assert len(self.info.waiting[fp]) == 2
            return response

        rsp1 = Response("http://url")

        async def rsp1_func():
            await _process_pending_io()
            _check_downloading(rsp1)

        async def rsp2_func():
            pytest.fail("it must cache rsp1 result and must not try to redownload")

        req1 = Request("http://url", meta={"response": rsp1_func})
        req2 = Request(req1.url, meta={"response": rsp2_func})
        item = {"requests": [req1, req2]}
        new_item = await self.pipe.process_item(item)
        assert new_item["results"] == [(True, {}), (True, {})]

    @pytest.mark.parametrize(
        ("size", "prefixed", "cached_urls"),
        [
            (2, False, ["http://url1", "http://url3"]),
            (2, True, ["http://url1", "http://url3"]),
            (0, False, []),
            (-1, False, ["http://url2", "http://url1", "http://url3"]),
        ],
    )
    @coroutine_test
    async def test_cache_size(self, size: int, prefixed: bool, cached_urls: list[str]):
        key = "MEDIA_CACHE_SIZE"
        if prefixed:
            key = f"{self.pipeline_class.__name__.upper()}_{key}"
        crawler = get_crawler(DefaultSpider, {key: size})
        crawler.spider = crawler._create_spider()
        crawler.engine = MagicMock(download_async=mocked_download_func)
        pipe = build_from_crawler(self.pipeline_class, crawler)
        pipe.open_spider()
        for url in ["http://url1", "http://url2", "http://url1", "http://url3"]:
            req = Request(url, meta={"response": Response(url)})
            await pipe.process_item({"requests": req})
        assert list(pipe.spiderinfo.downloaded) == [
            self.fingerprint(Request(url)) for url in cached_urls
        ]

    @coroutine_test
    async def test_use_media_to_download_result(self):
        req = Request("http://url", meta={"result": "ITSME"})
        item = {"requests": req}
        new_item = await self.pipe.process_item(item)
        assert new_item["results"] == [(True, "ITSME")]
        assert self.mocked_pipe._mockcalled == [
            "get_media_requests",
            "media_to_download",
            "item_completed",
        ]

    def test_key_for_pipe(self):
        assert (
            self.pipe._key_for_pipe("IMAGES", base_class_name="MediaPipeline")
            == "MOCKEDMEDIAPIPELINE_IMAGES"
        )


class TestAsyncMediaDownloaded(TestMediaPipeline):
    pipeline_class = AsyncMediaDownloadedPipeline

    def test_key_for_pipe(self):
        assert (
            self.pipe._key_for_pipe("IMAGES", base_class_name="MediaPipeline")
            == "ASYNCMEDIADOWNLOADEDPIPELINE_IMAGES"
        )


class TestMediaPipelineAllowRedirectSettings:
    def _assert_redirect_statuses_not_handled(
        self, pipeline_class: type[MediaPipeline], settings: dict[str, Any]
    ) -> None:
        pipe = pipeline_class(crawler=get_crawler(None, settings))
        request = Request("http://url")
        pipe._modify_media_request(request)

        assert "handle_httpstatus_list" in request.meta
        for status, check in [
            (200, True),
            # 201 is not in REDIRECT_HTTP_CODES by default
            (201, True),
            # These are the status codes we want
            # the downloader to handle itself
            (301, False),
            (302, False),
            (303, False),
            (307, False),
            (308, False),
            # we still want to get 4xx and 5xx
            (400, True),
            (404, True),
            (500, True),
        ]:
            if check:
                assert status in request.meta["handle_httpstatus_list"]
            else:
                assert status not in request.meta["handle_httpstatus_list"]

    def test_subclass_standard_setting(self):
        self._assert_redirect_statuses_not_handled(
            UserDefinedPipeline, {"MEDIA_ALLOW_REDIRECTS": True}
        )

    def test_subclass_specific_setting(self):
        self._assert_redirect_statuses_not_handled(
            UserDefinedPipeline, {"USERDEFINEDPIPELINE_MEDIA_ALLOW_REDIRECTS": True}
        )

    def test_custom_redirect_http_codes(self):
        """Statuses added to REDIRECT_HTTP_CODES are also left to
        RedirectMiddleware."""
        crawler = get_crawler(
            None,
            {
                "MEDIA_ALLOW_REDIRECTS": True,
                "REDIRECT_HTTP_CODES": [201, 301, 302, 303, 307, 308],
            },
        )
        request = Request("http://url")
        UserDefinedPipeline(crawler=crawler)._modify_media_request(request)

        assert 201 not in request.meta["handle_httpstatus_list"]
        assert 302 not in request.meta["handle_httpstatus_list"]
        assert 200 in request.meta["handle_httpstatus_list"]


class TestBuildFromCrawler:
    def setup_method(self):
        self.crawler = get_crawler(None, {"FILES_STORE": "/foo"})

    def test_simple(self):
        class Pipeline(UserDefinedPipeline):
            pass

        pipe = build_from_crawler(Pipeline, self.crawler)
        assert pipe.crawler == self.crawler
        assert pipe._fingerprinter

    def test_has_from_crawler_and_init(self):
        class Pipeline(UserDefinedPipeline):
            _from_crawler_called = False

            def __init__(self, store_uri, settings, *, crawler):
                super().__init__(crawler=crawler)
                self._init_called = True

            @classmethod
            def from_crawler(cls, crawler: Crawler) -> Pipeline:
                settings = crawler.settings
                store_uri = settings["FILES_STORE"]
                o = cls(store_uri, settings=settings, crawler=crawler)
                o._from_crawler_called = True
                return o

        pipe = build_from_crawler(Pipeline, self.crawler)
        assert pipe.crawler == self.crawler
        assert pipe._fingerprinter
        assert pipe._from_crawler_called
        assert pipe._init_called

    def test_has_from_crawler(self):
        class Pipeline(UserDefinedPipeline):
            _from_crawler_called = False
            store_uri: str

            @classmethod
            def from_crawler(cls, crawler: Crawler) -> Pipeline:
                settings = crawler.settings
                o = super().from_crawler(crawler)
                o._from_crawler_called = True
                o.store_uri = settings["FILES_STORE"]
                return o

        pipe = build_from_crawler(Pipeline, self.crawler)
        assert pipe.crawler == self.crawler
        assert pipe._fingerprinter
        assert pipe._from_crawler_called


class MediaFailedNonePipeline(MockedMediaPipeline):
    def media_failed(  # type: ignore[override]
        self, failure: Failure, request: Request, info: MediaPipeline.SpiderInfo
    ) -> None:
        self._mockcalled.append("media_failed")


class TestMediaFailedNone(TestBaseMediaPipeline):
    """Test what happens when media_failed() neither raises an exception nor
    returns a failure."""

    pipeline_class = MediaFailedNonePipeline

    @coroutine_test
    async def test_result_none(self):
        req = Request("http://url1", meta={"response": Exception("foo")})
        new_item = await self.pipe.process_item({"requests": req})
        assert new_item["results"] == [(True, None)]
        assert self.mocked_pipe._mockcalled == [
            "get_media_requests",
            "media_to_download",
            "media_failed",
            "item_completed",
        ]


class MediaFailedFailurePipeline(MockedMediaPipeline):
    def media_failed(
        self, failure: Failure, request: Request, info: MediaPipeline.SpiderInfo
    ) -> Failure:
        self._mockcalled.append("media_failed")
        return failure  # deprecated


class TestMediaFailedFailure(TestBaseMediaPipeline):
    """Test that media_failed() can return a failure instead of raising."""

    pipeline_class = MediaFailedFailurePipeline

    def _errback(self, result):
        self.mocked_pipe._mockcalled.append("request_errback")
        return result

    @coroutine_test
    async def test_result_failure(self):
        self.pipe.LOG_FAILED_RESULTS = False
        exc = Exception("foo")
        req = Request(
            "http://url1",
            meta={"response": exc},
            errback=self._errback,
        )
        item = {"requests": req}
        with pytest.warns(
            ScrapyDeprecationWarning, match="media_failed returned a Failure instance"
        ):
            new_item = await self.pipe.process_item(item)
        assert len(new_item["results"]) == 1
        assert new_item["results"][0][0] is False
        assert isinstance(new_item["results"][0][1], Failure)
        assert new_item["results"][0][1].value == exc
        assert self.mocked_pipe._mockcalled == [
            "get_media_requests",
            "media_to_download",
            "media_failed",
            "request_errback",
            "item_completed",
        ]


class _ControlledDownload:
    """A stand-in for ``engine.download_async`` whose completion and
    cancellation are driven by the test through
    :class:`~twisted.internet.defer.Deferred` objects, so the behavior is
    identical with either reactor implementation.

    Each call parks on a pending Deferred. When it is cancelled the coroutine
    either propagates the cancellation or, when *survive_cancellation* is set,
    moves onto a second Deferred that the test can resolve afterwards, modeling
    a download whose result arrives after the cancellation.
    """

    def __init__(self, *, survive_cancellation: bool = False) -> None:
        self.pending: list[Deferred[Response]] = []
        self.cancelled: list[Deferred[Response]] = []
        # Deferreds awaited by coroutines that survived their cancellation.
        self.survivors: list[Deferred[Response]] = []
        self.calls = 0
        self._survive = survive_cancellation

    async def __call__(self, request: Request) -> Response:
        self.calls += 1
        first: Deferred[Response] = Deferred()
        self.pending.append(first)
        try:
            return await maybe_deferred_to_future(first)
        except (CancelledError, asyncio.CancelledError):
            self.cancelled.append(first)
            if not self._survive:
                raise
            late: Deferred[Response] = Deferred()
            self.survivors.append(late)
            return await maybe_deferred_to_future(late)


class _CancellationPipeline(MockedMediaPipeline):
    """Pipeline whose item_completed records that it ran, so a cancelled item
    can be distinguished from one that completed normally."""

    def __init__(self, *args: Any, crawler: Crawler, **kwargs: Any) -> None:
        super().__init__(*args, crawler=crawler, **kwargs)
        self.comp_items: list[Any] = []

    def item_completed(
        self,
        results: list[FileInfoOrError],
        item: Any,
        info: MediaPipeline.SpiderInfo,
    ) -> Any:
        self.comp_items.append(item)
        return super().item_completed(results, item, info)


def _cancellation_request(url: str = "http://example.com/file") -> Request:
    return Request(url)


@pytest.mark.only_asyncio
class TestMediaCancellation:
    """Cancelling the Deferred returned by process_item() must only end that
    item; other items sharing the same media download keep waiting and get
    their result, independently of which item started the download."""

    def setup_method(self) -> None:
        crawler = get_crawler(DefaultSpider)
        crawler.spider = crawler._create_spider()
        self.download = _ControlledDownload()
        crawler.engine = MagicMock(download_async=self.download)
        self.pipe = build_from_crawler(_CancellationPipeline, crawler)
        self.pipe.open_spider()
        self.info = self.pipe.spiderinfo
        self.fingerprint = crawler.request_fingerprinter.fingerprint

    def teardown_method(self) -> None:
        for name, signal in vars(signals).items():
            if not name.startswith("_"):
                disconnect_all(signal)

    def _start_item(self, *requests: Request) -> Deferred[Any]:
        # A Deferred keeps the same cancellation semantics on every reactor.
        return deferred_from_coro(self.pipe.process_item({"requests": list(requests)}))

    async def _flush(self) -> None:
        # Let the download coroutine register its Deferred and the waiters
        # park on theirs.
        for _ in range(5):
            await sleep(0)

    async def _assert_cancelled(self, deferred: Deferred[Any]) -> None:
        # The concrete CancelledError class depends on the reactor.
        with pytest.raises((CancelledError, asyncio.CancelledError)):
            await maybe_deferred_to_future(deferred)

    @coroutine_test
    async def test_cancel_first_waiter_keeps_shared_download(self) -> None:
        req1 = _cancellation_request()
        req2 = Request(req1.url)
        fp = self.fingerprint(req1)

        first = self._start_item(req1)
        await self._flush()
        second = self._start_item(req2)
        await self._flush()
        assert fp in self.info.downloading
        assert len(self.info.waiting[fp]) == 2

        first.cancel()
        await self._assert_cancelled(first)
        # The download stays alive for the second item; only its waiter
        # remains registered.
        assert fp in self.info.downloading
        assert len(self.info.waiting[fp]) == 1
        assert not self.download.cancelled

        self.download.pending[0].callback(Response(req1.url))
        item = await maybe_deferred_to_future(second)
        assert item["results"] == [(True, {})]
        assert fp in self.info.downloaded
        assert self.pipe.comp_items == [item]

    @coroutine_test
    async def test_cancel_second_waiter_keeps_shared_download(self) -> None:
        # Initiation order must not matter: cancelling a waiter that joined a
        # running download leaves the originating item intact.
        req1 = _cancellation_request()
        req2 = Request(req1.url)
        fp = self.fingerprint(req1)

        first = self._start_item(req1)
        await self._flush()
        second = self._start_item(req2)
        await self._flush()

        second.cancel()
        await self._assert_cancelled(second)
        assert fp in self.info.downloading
        assert len(self.info.waiting[fp]) == 1
        assert not self.download.cancelled

        self.download.pending[0].callback(Response(req1.url))
        item = await maybe_deferred_to_future(first)
        assert item["results"] == [(True, {})]

    @coroutine_test
    async def test_cancel_only_waiter_cancels_download_and_frees_state(
        self,
    ) -> None:
        req = _cancellation_request()
        fp = self.fingerprint(req)
        task = self._start_item(req)
        await self._flush()
        assert self.download.pending

        task.cancel()
        await self._assert_cancelled(task)
        await self._flush()

        assert self.download.cancelled
        assert fp not in self.info.downloading
        assert not self.info.download_states
        assert not self.info.waiting
        # A cancelled download is never cached as a failure.
        assert fp not in self.info.downloaded
        assert self.pipe.comp_items == []

    @coroutine_test
    async def test_retry_after_full_cancellation_is_clean(self) -> None:
        # Once every waiter was cancelled, a new item for the same URL starts
        # a fresh download instead of inheriting the cancelled state.
        req = _cancellation_request()
        fp = self.fingerprint(req)
        task = self._start_item(req)
        await self._flush()
        task.cancel()
        await self._flush()
        assert self.download.calls == 1

        retry = self._start_item(Request(req.url))
        await self._flush()
        assert self.download.calls == 2
        assert fp in self.info.downloading
        self.download.pending[1].callback(Response(req.url))
        item = await maybe_deferred_to_future(retry)
        assert item["results"] == [(True, {})]

    @coroutine_test
    async def test_late_result_does_not_poison_new_download(self) -> None:
        # A download whose last waiter was cancelled is detached from the
        # state. A result arriving late must neither be cached nor reach the
        # waiters of a newer download for the same fingerprint.
        self.download = _ControlledDownload(survive_cancellation=True)
        self.pipe.crawler.engine = MagicMock(download_async=self.download)

        req = _cancellation_request()
        fp = self.fingerprint(req)
        task = self._start_item(req)
        await self._flush()
        stale_deferred = self.download.pending[0]
        task.cancel()
        await self._flush()
        # The cancelled coroutine stays alive, waiting on a new Deferred.
        assert len(self.download.survivors) == 1
        survivor = self.download.survivors[0]

        retry = self._start_item(Request(req.url))
        await self._flush()
        # The retry owns a fresh download, unrelated to the surviving one.
        assert self.download.pending[1] is not stale_deferred
        assert self.download.pending[1] is not survivor
        assert fp in self.info.downloading

        # The surviving download reports a failure too late.
        survivor.errback(Failure(RuntimeError("late")))
        await self._flush()
        assert fp not in self.info.downloaded

        self.download.pending[1].callback(Response(req.url))
        item = await maybe_deferred_to_future(retry)
        assert item["results"] == [(True, {})]

    @coroutine_test
    async def test_duplicate_urls_in_one_item_cancel_together(self) -> None:
        # One item may request the same URL more than once. Cancelling its
        # process_item() reaches every one of its waiters, while a different
        # item sharing the download survives.
        req1 = _cancellation_request()
        duplicated = Request(req1.url)
        other = Request(req1.url)
        fp = self.fingerprint(req1)

        duplicated_item = self._start_item(req1, duplicated)
        await self._flush()
        other_item = self._start_item(other)
        await self._flush()
        # Two waiters belong to the duplicated item, one to the other item.
        assert len(self.info.waiting[fp]) == 3

        duplicated_item.cancel()
        await self._assert_cancelled(duplicated_item)
        assert fp in self.info.downloading
        assert len(self.info.waiting[fp]) == 1
        assert not self.download.cancelled

        self.download.pending[0].callback(Response(req1.url))
        item = await maybe_deferred_to_future(other_item)
        assert item["results"] == [(True, {})]

    @coroutine_test
    async def test_cancelled_request_skips_errback(self) -> None:
        calls: list[str] = []

        def errback(failure: Failure) -> FileInfo:
            calls.append("errback")
            return cast("FileInfo", {"recovered": True})

        req = Request("http://example.com/file", errback=errback)
        task = self._start_item(req)
        await self._flush()
        task.cancel()
        await self._assert_cancelled(task)
        assert calls == []

    @coroutine_test
    async def test_uncancelled_failure_still_calls_errback_and_caches(
        self,
    ) -> None:
        # The ordinary failure path and its errback behavior are unchanged.
        exc = RuntimeError("boom")
        req = Request(
            "http://example.com/file",
            errback=lambda failure: cast("FileInfo", {"recovered": failure.value}),
        )
        task = self._start_item(req)
        await self._flush()
        self.download.pending[0].errback(Failure(exc))
        item = await maybe_deferred_to_future(task)
        assert item["results"] == [(True, {"recovered": exc})]
        assert self.fingerprint(req) in self.info.downloaded

    @coroutine_test
    async def test_distinct_crawlers_do_not_share_state(self) -> None:
        other_crawler = get_crawler(DefaultSpider)
        other_crawler.spider = other_crawler._create_spider()
        other_download = _ControlledDownload()
        other_crawler.engine = MagicMock(download_async=other_download)
        other_pipe = build_from_crawler(_CancellationPipeline, other_crawler)
        other_pipe.open_spider()

        req = _cancellation_request()
        task = self._start_item(req)
        await self._flush()
        task.cancel()
        await self._flush()

        # The other crawler owns a separate SpiderInfo and is unaffected.
        other_info = other_pipe.spiderinfo
        assert other_info is not self.info
        assert not other_info.download_states
        other_item = deferred_from_coro(
            other_pipe.process_item({"requests": [Request(req.url)]})
        )
        await self._flush()
        assert other_download.calls == 1
        other_download.pending[0].callback(Response(req.url))
        result = await maybe_deferred_to_future(other_item)
        assert result["results"] == [(True, {})]


@pytest.mark.only_not_asyncio
class TestMediaCancellationNonAsyncio:
    """Cancellation behavior for the deprecated non-asyncio reactor path,
    where twisted's CancelledError is an Exception and DeferredList does not
    propagate cancellations on its own."""

    def setup_method(self) -> None:
        crawler = get_crawler(DefaultSpider)
        crawler.spider = crawler._create_spider()
        self.download = _ControlledDownload()
        crawler.engine = MagicMock(download_async=self.download)
        self.pipe = build_from_crawler(_CancellationPipeline, crawler)
        self.pipe.open_spider()
        self.info = self.pipe.spiderinfo
        self.fingerprint = crawler.request_fingerprinter.fingerprint

    def teardown_method(self) -> None:
        for name, signal in vars(signals).items():
            if not name.startswith("_"):
                disconnect_all(signal)

    async def _flush(self) -> None:
        for _ in range(5):
            await sleep(0)

    @coroutine_test
    async def test_process_item_cancel_ends_with_cancelled_error(self) -> None:
        req = _cancellation_request()
        fp = self.fingerprint(req)
        item_deferred = deferred_from_coro(self.pipe.process_item({"requests": [req]}))
        await self._flush()

        item_deferred.cancel()
        with pytest.raises(CancelledError):
            await item_deferred

        await self._flush()
        # The shared download was cancelled and no state or failure leaked.
        assert self.download.cancelled
        assert fp not in self.info.downloading
        assert not self.info.download_states
        assert fp not in self.info.downloaded
        assert self.pipe.comp_items == []

    @coroutine_test
    async def test_shared_download_survives_first_cancellation(self) -> None:
        req1 = _cancellation_request()
        req2 = Request(req1.url)
        fp = self.fingerprint(req1)

        first = deferred_from_coro(self.pipe.process_item({"requests": [req1]}))
        await self._flush()
        second = deferred_from_coro(self.pipe.process_item({"requests": [req2]}))
        await self._flush()

        first.cancel()
        with pytest.raises(CancelledError):
            await first
        assert fp in self.info.downloading
        assert not self.download.cancelled

        self.download.pending[0].callback(Response(req1.url))
        item = await maybe_deferred_to_future(second)
        assert item["results"] == [(True, {})]
