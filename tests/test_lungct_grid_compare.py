"""The optional page selector must preserve standard HTTP behavior."""
import http.client
from pathlib import Path
import threading
import pytest
from scripts.visualization.lungct_viewer.serve import Handler, Server


@pytest.mark.parametrize('viewer,title',[(None,'Registration viewer'),('standard','Registration viewer'),('grid-compare','Three grid constructions')])
def test_grid_page_selection_is_readonly_and_standard_is_default(tmp_path,viewer,title):
    server=Server(('127.0.0.1',0),Handler)
    server.web_root=Path(__file__).parents[1]/'scripts/visualization/lungct_viewer/web'
    server.bundle_root=tmp_path
    if viewer is not None:server.viewer=viewer
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    conn=http.client.HTTPConnection(*server.server_address)
    try:
        conn.request('GET','/');r=conn.getresponse()
        assert r.status==200 and title in r.read().decode()
        conn.request('GET','/grid_compare/cache.js');r=conn.getresponse()
        assert r.status==200 and b'buildGridCache' in r.read()
        conn.request('GET','/grid_compare/../../../AGENTS.md');r=conn.getresponse()
        assert r.status==404;r.read()
        conn.request('POST','/');r=conn.getresponse();assert r.status==501;r.read()
    finally:
        conn.close();server.shutdown();server.server_close();thread.join()
