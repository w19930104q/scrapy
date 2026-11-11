from __future__ import annotations

import gzip
import json
import os
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any

import pytest

import scrapy
from scrapy.exceptions import ScrapyDeprecationWarning
from scrapy.exporters import JsonLinesItemExporter
from scrapy.extensions.feedexport import FeedExporter, FileFeedStorage
from scrapy.utils.test import get_crawler
from tests.utils.decorators import coroutine_test
from tests.utils.feedexport import (
    PARSERS,
    crawl_items,
    csv_header,
    path_to_url,
    printf_escape,
    unique_path,
)

if TYPE_CHECKING:
    from scrapy.crawler import Crawler
    from tests.mockserver.http import MockServer


def _file_uri(path: Path) -> str:
    return printf_escape(path_to_url(path))


def _atomic_feed(
    path: Path,
    *,
    fmt: str = "jl",
    uri_type: str = "path",
    **options: Any,
) -> dict[str, Any]:
    uri = str(path) if uri_type == "path" else _file_uri(path)
    return {uri: {"format": fmt, "mode": "overwrite", "atomic": True, **options}}


def _names(directory: Path) -> list[str]:
    """Return the names of the entries of *directory*, including hidden ones."""
    return sorted(entry.name for entry in directory.iterdir())


class _InitFailingJsonLinesItemExporter(JsonLinesItemExporter):
    # pylint: disable-next=super-init-not-called
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError("init failed")


class _ExportFailingJsonLinesItemExporter(JsonLinesItemExporter):
    def export_item(self, item: Any) -> None:
        raise RuntimeError("export failed")


class _FinishFailingJsonLinesItemExporter(JsonLinesItemExporter):
    def finish_exporting(self) -> None:
        raise RuntimeError("finish failed")


class _NthItemFailingJsonLinesItemExporter(JsonLinesItemExporter):
    fail_at = 3
    calls = 0

    def export_item(self, item: Any) -> None:
        type(self).calls += 1
        if type(self).calls == self.fail_at:
            raise RuntimeError("export failed")
        super().export_item(item)


class _AsyncJsonLinesItemExporter(JsonLinesItemExporter):
    async def start_exporting(self) -> None:
        super().start_exporting()

    async def export_item(self, item: Any) -> None:  # type: ignore[override]
        JsonLinesItemExporter.export_item(self, item)

    async def finish_exporting(self) -> None:
        super().finish_exporting()


class _FailingClosePlugin:
    def __init__(self, file: IO[bytes], feed_options: dict[str, Any]) -> None:
        self.file = file

    def write(self, data: bytes) -> int:
        return self.file.write(data)

    def close(self) -> None:
        raise RuntimeError("close failed")


class _FailingFileFeedStorage(FileFeedStorage):
    """Local file storage whose ``store()`` fails once, for the *fail_at*-th
    call."""

    fail_at = 2
    stores = 0
    error: type[Exception] = RuntimeError

    def __init__(self, uri: str, *, feed_options: dict[str, Any] | None = None) -> None:
        super().__init__(uri.removeprefix("failing:"), feed_options=feed_options)

    def store(self, file: IO[bytes]) -> None:
        type(self).stores += 1
        if type(self).stores == self.fail_at:
            raise self.error("storing failed")
        super().store(file)


class TestAtomicValidation:
    @pytest.mark.parametrize(
        "uri",
        [
            "stdout:",
            "ftp://user:pass@ftp.example.com/items.json",
        ],
    )
    def test_non_file_storage(self, uri: str) -> None:
        settings = {
            "FEEDS": {uri: {"format": "json", "mode": "overwrite", "atomic": True}}
        }
        with pytest.raises(ValueError, match="only supported for local filesystem"):
            get_crawler(settings_dict=settings)

    @pytest.mark.parametrize("mode", ["append", "create"])
    def test_non_overwrite_mode(self, mode: str) -> None:
        settings = {
            "FEEDS": {
                "file:///tmp/items.json": {
                    "format": "json",
                    "mode": mode,
                    "atomic": True,
                }
            }
        }
        with pytest.raises(ValueError, match="requires the 'overwrite' feed mode"):
            get_crawler(settings_dict=settings)

    def test_unset_mode(self) -> None:
        settings = {
            "FEEDS": {"file:///tmp/items.json": {"format": "json", "atomic": True}},
            "FEED_MODE": None,
        }
        with (
            pytest.warns(ScrapyDeprecationWarning, match="FEED_MODE"),
            pytest.raises(ValueError, match="requires the 'overwrite' feed mode"),
        ):
            get_crawler(settings_dict=settings)

    def test_legacy_overwrite(self) -> None:
        settings: dict[str, Any] = {
            "FEEDS": {"file:///tmp/items.json": {"format": "json", "atomic": True}},
            "FEED_MODE": "append",
        }
        settings["FEEDS"]["file:///tmp/items.json"]["overwrite"] = True
        with pytest.warns(ScrapyDeprecationWarning, match="overwrite feed option"):
            crawler = get_crawler(settings_dict=settings)
        exporter = next(
            middleware
            for middleware in crawler.extensions.middlewares
            if isinstance(middleware, FeedExporter)
        )
        assert exporter.feeds["file:///tmp/items.json"]["mode"] == "overwrite"


class TestAtomicCommit:
    @pytest.mark.parametrize("uri_type", ["path", "file_uri"])
    @coroutine_test
    async def test_single_batch(
        self, mockserver: MockServer, tmp_path: Path, uri_type: str
    ) -> None:
        path = unique_path(tmp_path)
        crawler = await crawl_items(
            mockserver,
            [{"foo": "bar1"}, {"foo": "bar2"}],
            {"FEEDS": _atomic_feed(path, uri_type=uri_type)},
        )
        assert path.read_bytes() == b'{"foo": "bar1"}\n{"foo": "bar2"}\n'
        assert _names(tmp_path) == [path.name]
        assert crawler.stats
        stats = crawler.stats.get_stats()
        assert stats.get("feedexport/success_count/FileFeedStorage") == 1
        assert "feedexport/failed_count/FileFeedStorage" not in stats

    @pytest.mark.parametrize("fmt", ["json", "xml", "csv"])
    @coroutine_test
    async def test_same_output_as_non_atomic(
        self, mockserver: MockServer, tmp_path: Path, fmt: str
    ) -> None:
        atomic_path = unique_path(tmp_path)
        regular_path = unique_path(tmp_path)
        settings = {
            "FEEDS": {
                **_atomic_feed(atomic_path, fmt=fmt),
                _file_uri(regular_path): {"format": fmt, "mode": "overwrite"},
            }
        }
        await crawl_items(
            mockserver,
            [
                {"foo": "bar1", "egg": "spam1"},
                {"foo": "bar2", "egg": "spam2", "baz": "quux2"},
            ],
            settings,
        )
        assert atomic_path.read_bytes() == regular_path.read_bytes()

    @pytest.mark.parametrize("fmt", ["json", "xml", "csv"])
    @coroutine_test
    async def test_gzip(self, mockserver: MockServer, tmp_path: Path, fmt: str) -> None:
        path = unique_path(tmp_path)
        settings = {
            "FEEDS": _atomic_feed(
                path,
                fmt=fmt,
                postprocessing=["scrapy.extensions.postprocessing.GzipPlugin"],
            )
        }
        await crawl_items(mockserver, [{"foo": "bar1"}, {"foo": "bar2"}], settings)
        data = gzip.decompress(path.read_bytes())
        if fmt == "csv":
            assert csv_header(data) == ["foo"]
        assert PARSERS[fmt](data) == [{"foo": "bar1"}, {"foo": "bar2"}]
        assert _names(tmp_path) == [path.name]

    @coroutine_test
    async def test_gzip_batches(self, mockserver: MockServer, tmp_path: Path) -> None:
        directory = unique_path(tmp_path)
        settings = {
            "FEEDS": _atomic_feed(
                directory / "%(batch_id)d.json",
                fmt="json",
                postprocessing=["scrapy.extensions.postprocessing.GzipPlugin"],
            ),
            "FEED_EXPORT_BATCH_ITEM_COUNT": 1,
        }
        await crawl_items(mockserver, [{"foo": "bar1"}, {"foo": "bar2"}], settings)
        assert _names(directory) == ["1.json", "2.json"]
        assert [
            json.loads(gzip.decompress((directory / name).read_bytes()))
            for name in ("1.json", "2.json")
        ] == [[{"foo": "bar1"}], [{"foo": "bar2"}]]

    @pytest.mark.parametrize(
        "exporter",
        [None, _AsyncJsonLinesItemExporter],
        ids=["sync-exporter", "async-exporter"],
    )
    @coroutine_test
    async def test_exporters(
        self,
        mockserver: MockServer,
        tmp_path: Path,
        exporter: type[JsonLinesItemExporter] | None,
    ) -> None:
        path = unique_path(tmp_path)
        settings: dict[str, Any] = {"FEEDS": _atomic_feed(path)}
        if exporter is not None:
            settings["FEED_EXPORTERS"] = {"jl": exporter}
        crawler = await crawl_items(
            mockserver, [{"foo": "bar1"}, {"foo": "bar2"}], settings
        )
        assert path.read_bytes() == b'{"foo": "bar1"}\n{"foo": "bar2"}\n'
        assert crawler.stats
        assert crawler.stats.get_value("feedexport/success_count/FileFeedStorage") == 1

    @coroutine_test
    async def test_multiple_feeds_and_batches(
        self, mockserver: MockServer, tmp_path: Path
    ) -> None:
        json_dir = unique_path(tmp_path)
        csv_dir = unique_path(tmp_path)
        settings = {
            "FEEDS": {
                **_atomic_feed(json_dir / "%(batch_id)d.jsonl", fmt="jsonlines"),
                **_atomic_feed(csv_dir / "%(batch_id)d.csv", fmt="csv"),
            },
            "FEED_EXPORT_BATCH_ITEM_COUNT": 1,
        }
        crawler = await crawl_items(
            mockserver,
            [
                {"foo": "bar1", "egg": "spam1"},
                {"foo": "bar2", "egg": "spam2"},
                {"foo": "bar3", "egg": "spam3"},
            ],
            settings,
        )
        assert _names(json_dir) == ["1.jsonl", "2.jsonl", "3.jsonl"]
        assert _names(csv_dir) == ["1.csv", "2.csv", "3.csv"]
        assert [
            PARSERS["jsonlines"]((json_dir / name).read_bytes())
            for name in _names(json_dir)
        ] == [
            [{"foo": "bar1", "egg": "spam1"}],
            [{"foo": "bar2", "egg": "spam2"}],
            [{"foo": "bar3", "egg": "spam3"}],
        ]
        assert [
            csv_header((csv_dir / name).read_bytes()) for name in _names(csv_dir)
        ] == [["foo", "egg"]] * 3
        assert crawler.stats
        stats = crawler.stats.get_stats()
        assert stats.get("feedexport/success_count/FileFeedStorage") == 6

    @pytest.mark.parametrize("store_empty", [True, False], ids=["store", "skip"])
    @coroutine_test
    async def test_empty_feed(
        self, mockserver: MockServer, tmp_path: Path, store_empty: bool
    ) -> None:
        path = unique_path(tmp_path)
        settings = {
            "FEEDS": _atomic_feed(path),
            "FEED_STORE_EMPTY": store_empty,
        }
        crawler = await crawl_items(mockserver, [], settings)
        if store_empty:
            assert path.read_bytes() == b""
            assert crawler.stats
            assert (
                crawler.stats.get_value("feedexport/success_count/FileFeedStorage") == 1
            )
        else:
            assert not path.exists()
        assert _names(tmp_path) == ([path.name] if store_empty else [])

    @coroutine_test
    async def test_legacy_overwrite(
        self, mockserver: MockServer, tmp_path: Path
    ) -> None:
        path = unique_path(tmp_path)
        settings = {
            "FEEDS": {str(path): {"format": "jl", "atomic": True, "overwrite": True}}
        }
        with pytest.warns(ScrapyDeprecationWarning, match="overwrite feed option"):
            await crawl_items(mockserver, [{"foo": "bar"}], settings)
        assert path.read_bytes() == b'{"foo": "bar"}\n'
        assert _names(tmp_path) == [path.name]

    @coroutine_test
    async def test_target_is_not_exposed_before_storing(
        self, mockserver: MockServer, tmp_path: Path
    ) -> None:
        """Neither a new target nor the previous content of an existing target
        is modified while the batch is being written."""
        new_path = unique_path(tmp_path)
        existing_path = tmp_path / "existing.jl"
        existing_path.write_bytes(b"old content")
        new_target_exists: list[bool] = []
        existing_content: list[bytes] = []
        mockserver_url = mockserver.url("/")

        class TestSpider(scrapy.Spider):
            name = "testspider"
            start_urls = [mockserver_url]

            def parse(self, response):
                yield {"foo": "bar1"}
                yield scrapy.Request(
                    mockserver_url, callback=self.parse2, dont_filter=True
                )

            def parse2(self, response):
                new_target_exists.append(new_path.exists())
                existing_content.append(existing_path.read_bytes())
                yield {"foo": "bar2"}

        settings = {
            "FEEDS": {
                **_atomic_feed(new_path),
                **_atomic_feed(existing_path),
            }
        }
        crawler = get_crawler(TestSpider, settings)
        await crawler.crawl_async()

        assert new_target_exists == [False]
        assert existing_content == [b"old content"]
        assert new_path.read_bytes() == b'{"foo": "bar1"}\n{"foo": "bar2"}\n'
        assert existing_path.read_bytes() == b'{"foo": "bar1"}\n{"foo": "bar2"}\n'


class TestAtomicFailures:
    """A failure at any stage keeps the previous target, removes the temporary
    file, counts one failure, and does not count a success."""

    @staticmethod
    def _assert_failed(
        directory: Path,
        crawler: Crawler,
        *,
        path: Path,
        expected: bytes | None,
    ) -> None:
        if expected is None:
            assert not path.exists()
            assert _names(directory) == []
        else:
            assert path.read_bytes() == expected
            assert _names(directory) == [path.name]
        assert crawler.stats
        stats = crawler.stats.get_stats()
        assert stats.get("feedexport/failed_count/FileFeedStorage") == 1
        assert "feedexport/success_count/FileFeedStorage" not in stats
        assert stats.get("finish_reason") == "finished"

    @pytest.mark.parametrize(
        ("exporter", "engine_stage"),
        [
            pytest.param(_InitFailingJsonLinesItemExporter, "init", id="init"),
            pytest.param(_ExportFailingJsonLinesItemExporter, "write", id="write"),
            pytest.param(_FinishFailingJsonLinesItemExporter, "finish", id="finish"),
            pytest.param(None, "flush", id="flush"),
        ],
    )
    @coroutine_test
    async def test_exporter_stage_failure(
        self,
        mockserver: MockServer,
        tmp_path: Path,
        exporter: type[JsonLinesItemExporter] | None,
        engine_stage: str,
    ) -> None:
        path = tmp_path / "items.jl"
        path.write_bytes(b"old content")
        settings: dict[str, Any] = {"FEEDS": _atomic_feed(path)}
        if exporter is not None:
            settings["FEED_EXPORTERS"] = {"jl": exporter}
        if engine_stage == "flush":
            settings["FEEDS"][str(path)]["postprocessing"] = [_FailingClosePlugin]
        crawler = await crawl_items(mockserver, [{"foo": "bar"}], settings)
        self._assert_failed(tmp_path, crawler, path=path, expected=b"old content")

    @coroutine_test
    async def test_init_failure_new_target(
        self, mockserver: MockServer, tmp_path: Path
    ) -> None:
        """A failure before the target is first created does not expose a
        partial file."""
        path = unique_path(tmp_path)
        settings = {
            "FEEDS": _atomic_feed(path),
            "FEED_EXPORTERS": {"jl": _InitFailingJsonLinesItemExporter},
        }
        crawler = await crawl_items(mockserver, [{"foo": "bar"}], settings)
        self._assert_failed(tmp_path, crawler, path=path, expected=None)

    @coroutine_test
    async def test_empty_feed_init_failure(
        self, mockserver: MockServer, tmp_path: Path
    ) -> None:
        """An empty atomic feed that cannot be opened at close time is not
        stored either."""
        path = unique_path(tmp_path)
        settings = {
            "FEEDS": _atomic_feed(path),
            "FEED_STORE_EMPTY": True,
            "FEED_EXPORTERS": {"jl": _InitFailingJsonLinesItemExporter},
        }
        crawler = await crawl_items(mockserver, [], settings)
        self._assert_failed(tmp_path, crawler, path=path, expected=None)

    @coroutine_test
    async def test_replace_failure(
        self,
        mockserver: MockServer,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        path = tmp_path / "items.jl"
        path.write_bytes(b"old content")
        real_replace = os.replace

        def failing_replace(src: str, dst: str) -> None:
            if Path(dst) == path:
                raise OSError("replace failed")
            real_replace(src, dst)

        monkeypatch.setattr("os.replace", failing_replace)
        crawler = await crawl_items(
            mockserver, [{"foo": "bar"}], {"FEEDS": _atomic_feed(path)}
        )
        self._assert_failed(tmp_path, crawler, path=path, expected=b"old content")

    @pytest.mark.parametrize("error", [RuntimeError, FileExistsError])
    @coroutine_test
    async def test_failed_feed_discards_later_batches(
        self,
        mockserver: MockServer,
        tmp_path: Path,
        error: type[Exception],
    ) -> None:
        """Batches stored before the failure are kept; the failing batch and
        the later ones are not stored; other feeds are unaffected."""
        _FailingFileFeedStorage.stores = 0
        _FailingFileFeedStorage.error = error
        failing_dir = unique_path(tmp_path)
        regular_dir = unique_path(tmp_path)
        settings = {
            "FEEDS": {
                f"failing:{failing_dir / '%(batch_id)d.jsonl'}": {
                    "format": "jsonlines",
                    "mode": "overwrite",
                    "atomic": True,
                },
                **_atomic_feed(regular_dir / "%(batch_id)d.jsonl", fmt="jsonlines"),
            },
            "FEED_STORAGES": {"failing": _FailingFileFeedStorage},
            "FEED_EXPORT_BATCH_ITEM_COUNT": 1,
        }
        crawler = await crawl_items(
            mockserver,
            [{"foo": "bar1"}, {"foo": "bar2"}, {"foo": "bar3"}],
            settings,
        )
        assert _names(failing_dir) == ["1.jsonl"]
        assert _names(regular_dir) == ["1.jsonl", "2.jsonl", "3.jsonl"]
        assert crawler.stats
        stats = crawler.stats.get_stats()
        assert stats.get("feedexport/failed_count/_FailingFileFeedStorage") == 1
        assert stats.get("feedexport/success_count/_FailingFileFeedStorage") == 1
        assert stats.get("feedexport/success_count/FileFeedStorage") == 3
        assert stats.get("finish_reason") == "finished"

    @coroutine_test
    async def test_write_failure_discards_later_batches(
        self, mockserver: MockServer, tmp_path: Path
    ) -> None:
        """A write failure marks the feed as failed, so neither the batch being
        written nor the batch that was waiting to be stored is kept."""
        _NthItemFailingJsonLinesItemExporter.calls = 0
        directory = unique_path(tmp_path)
        settings = {
            "FEEDS": _atomic_feed(directory / "%(batch_id)d.jsonl", fmt="jsonlines"),
            "FEED_EXPORTERS": {"jsonlines": _NthItemFailingJsonLinesItemExporter},
            "FEED_EXPORT_BATCH_ITEM_COUNT": 2,
        }
        crawler = await crawl_items(
            mockserver,
            [{"foo": f"bar{index}"} for index in range(5)],
            settings,
        )
        assert _names(directory) == []
        assert crawler.stats
        stats = crawler.stats.get_stats()
        assert stats.get("feedexport/failed_count/FileFeedStorage") == 1
        assert "feedexport/success_count/FileFeedStorage" not in stats
        assert stats.get("finish_reason") == "finished"
