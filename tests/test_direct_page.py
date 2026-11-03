"""Exercise the real direct HTTP backend against a temporary loopback server."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hashlib
from pathlib import Path
import sys
import threading
import unittest
from unittest.mock import Mock, patch
import os

ROOT = Path(os.environ.get('HYPERSKILL_REPO', Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT))
from FlashOAgents.search_tools import (
    DIRECT_PAGE_FETCHES, read_page_direct, web_search_bing_rss, WebSearchTool,
)

HTML = b'<html><body><h1>Official fact</h1><p>Python 3.10</p><script>hidden_script()</script></body></html>'


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        status, kind, body = {
            '/html': (200, 'text/html; charset=utf-8', HTML),
            '/empty': (200, 'text/plain', b''),
            '/binary': (200, 'application/octet-stream', b'\x00\x01'),
            '/missing': (404, 'text/plain', b'missing'),
        }.get(self.path, (500, 'text/plain', b'unknown'))
        self.send_response(status)
        self.send_header('Content-Type', kind)
        self.end_headers()
        self.wfile.write(body)


class DirectPageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = 'http://127.0.0.1:' + str(cls.server.server_port)

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def setUp(self):
        DIRECT_PAGE_FETCHES.clear()

    def test_actual_http_html_text_and_digest(self):
        text = read_page_direct(self.base + '/html')
        self.assertIn('Python 3.10', text)
        self.assertNotIn('hidden_script', text)
        event = DIRECT_PAGE_FETCHES[-1]
        self.assertEqual(event['status'], 'ok')
        self.assertEqual(event['body_sha256'], hashlib.sha256(HTML).hexdigest())
        self.assertEqual(event['text'], text)
        self.assertNotIn('headers', event)

    def test_missing_page_is_failure(self):
        self.assertTrue(read_page_direct(self.base + '/missing').startswith('Error'))
        self.assertEqual(DIRECT_PAGE_FETCHES[-1]['http_status'], 404)
        self.assertEqual(DIRECT_PAGE_FETCHES[-1]['status'], 'error')

    def test_empty_page_not_success(self):
        self.assertTrue(read_page_direct(self.base + '/empty').startswith('Error'))
        self.assertEqual(DIRECT_PAGE_FETCHES[-1]['error_code'], 'empty_page')

    def test_binary_page_not_success(self):
        self.assertTrue(read_page_direct(self.base + '/binary').startswith('Error'))
        self.assertEqual(DIRECT_PAGE_FETCHES[-1]['error_code'], 'unsupported_content_type')

    def test_non_http_scheme_not_requested(self):
        self.assertTrue(read_page_direct('file:///etc/passwd').startswith('Error'))
        self.assertEqual(DIRECT_PAGE_FETCHES[-1]['error_code'], 'unsupported_scheme')


class SearchRssTests(unittest.TestCase):
    def response(self, body):
        result = Mock()
        result.content = body
        return result

    def test_rss_results_have_public_source_and_bounded_count(self):
        body = b'<rss><channel><item><title>A &amp; B</title><link>https://example.org/1</link><description>Fact</description></item><item><title>Second</title><link>https://example.org/2</link></item></channel></rss>'
        with patch('FlashOAgents.search_tools.requests.get', return_value=self.response(body)) as get:
            results, error = web_search_bing_rss('public query', serp_num=1)
        self.assertEqual(error, '')
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['title'], 'A & B')
        self.assertEqual(results[0]['link'], 'https://example.org/1')
        self.assertEqual(get.call_args.kwargs['params'], {'q': 'public query', 'format': 'rss'})

    def test_non_rss_or_empty_feed_is_not_search_success(self):
        for body in (b'<html>Challenge</html>', b'<rss><channel/></rss>', b'broken xml'):
            with self.subTest(body=body):
                with patch('FlashOAgents.search_tools.requests.get', return_value=self.response(body)):
                    results, error = web_search_bing_rss('query')
                self.assertEqual(results, [])
                self.assertTrue(error)

    def test_unknown_explicit_backend_never_falls_back(self):
        with patch.dict(os.environ, {'WEB_SEARCH_PROVIDER': 'unknown'}):
            with patch('FlashOAgents.search_tools.requests.get') as get:
                text = WebSearchTool().forward('query')
        self.assertTrue(text.startswith('Error'))
        get.assert_not_called()


if __name__ == '__main__':
    unittest.main(verbosity=2)
