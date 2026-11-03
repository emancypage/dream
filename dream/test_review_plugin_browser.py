"""Optional isolated-browser integration tests; never use the live Dream store."""
import json
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from test_character_judge import env
from test_character_review import judged
from test_review_plugin import app, server, PLUGIN
from character import storage

playwright = pytest.importorskip('playwright.sync_api')

HOST = '''<!doctype html><html><head><meta charset="utf-8"></head>
<body style="margin:0"><iframe title="Dream Review" src="/panel" style="width:100%;height:100vh;border:0"></iframe><script>
addEventListener('message',async e=>{const m=e.data;if(e.source!==document.querySelector('iframe').contentWindow||m?.jsonrpc!=='2.0'||m.id===undefined)return;
let result;if(m.method==='ui/initialize')result={hostInfo:{name:'fixture-host',version:'26.1028.0'},hostCapabilities:{serverTools:{}},hostContext:{platform:'desktop'}};
else if(m.method==='tools/call'){const r=await fetch('/rpc',{method:'POST',body:JSON.stringify(m.params)});result=await r.json();}
else result={};e.source.postMessage({jsonrpc:'2.0',id:m.id,result},'*');});</script></body></html>'''


@contextmanager
def fixture_host(server):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            body = (PLUGIN/'panel.html').read_bytes() if self.path == '/panel' else HOST.encode()
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            args = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            try:
                result = dict(structuredContent=server.call(args['name'], args['arguments']))
            except Exception as exc:
                result = dict(isError=True, content=[dict(type='text', text=str(exc))])
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())

        def log_message(self, *args):
            pass
    http = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=http.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{http.server_port}'
    finally:
        http.shutdown()
        http.server_close()
        thread.join()


def test_browser_pair_edit_and_separate_short_approval(server, env, monkeypatch):
    from types import SimpleNamespace
    from test_character_short import BODY
    j, r, cfg, conn, pair = env
    pair = storage.update_record(conn, 'character_pairs', dict(pair, revision=2, conflict=True,
        prompt_messages=[{'role':'user','content':'Jak sprawdzić, czy poprawka działa?'}],
        chosen='Sprawdź wynik na konkretnym przykładzie.\n\nUruchom test, który wcześniej pokazywał błąd. Napisz, co teraz przechodzi i czego jeszcze nie sprawdzono.',
        rejected='Wszystko powinno już działać.\n\nKod wygląda poprawnie i nie widzę powodów, żeby robić kolejne sprawdzenia.'),
        expected_sha=pair['content_sha256'])
    conn.commit()
    env = (j,r,cfg,conn,pair)
    j, r, cfg, conn, pair = judged(env, monkeypatch)
    monkeypatch.setattr(j.backend, 'generate', lambda *a, **k: SimpleNamespace(output=BODY))
    # HTTP callbacks use their own connections; this one is assertion-only.
    conn.close()
    monkeypatch.setattr(server.app, '_connect', type(server.app)._connect.__get__(server.app))
    with fixture_host(server) as url, playwright.sync_playwright() as driver:
        browser = driver.chromium.launch(headless=True)
        page = browser.new_page(viewport={'width':1440, 'height':1000})
        errors=[]
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.goto(url)
        frame=page.frame_locator('iframe')
        frame.get_by_role('button', name='Wybieram A', exact=True).wait_for()
        assert not frame.locator('#model').is_visible()
        page.set_viewport_size({'width':375, 'height':812})
        assert page.frames[1].evaluate('document.documentElement.scrollWidth <= innerWidth')
        page.set_viewport_size({'width':1440, 'height':1000})
        frame.get_by_role('button', name='Wybieram A', exact=True).click()
        assert frame.locator('#model').is_visible()
        frame.locator('#ack').check()
        frame.get_by_role('button', name='Wybieram B', exact=True).click()
        assert not frame.locator('#ack').is_checked()
        frame.get_by_role('button', name='Wybieram A', exact=True).click()
        assert not frame.locator('#ack').is_checked()
        frame.locator('#reason').fill('I prefer the direct answer.')
        page.reload()
        frame.get_by_role('button', name='Wybrana A', exact=True).wait_for()
        assert frame.locator('#reason').input_value() == 'I prefer the direct answer.'
        frame.get_by_role('button', name='Zapisz i przejdź dalej').click()
        assert frame.get_by_role('alert').inner_text() == 'Potwierdź widoczne zastrzeżenia przed zapisem.'
        frame.locator('#ack').check()
        frame.get_by_role('button', name='Zapisz i przejdź dalej').click()
        frame.get_by_role('heading', name='Nie ma teraz par do oceny.').wait_for()
        assert page.frames[1].evaluate("Object.keys(localStorage).filter(k=>k.startsWith('dream-review:p:')).length") == 0
        frame.get_by_role('tab', name='Skróty').click()
        frame.get_by_role('button', name='Popraw treść').wait_for()
        frame.get_by_role('button', name='Popraw treść').click()
        frame.locator('#field-preferred').fill('Check the actual result before claiming success.')
        page.reload()
        frame.get_by_role('button', name='Popraw treść').wait_for()
        frame.get_by_role('button', name='Popraw treść').click()
        assert frame.locator('#field-preferred').input_value() == 'Check the actual result before claiming success.'
        frame.get_by_role('button', name='Zapisz poprawiony szkic').click()
        frame.get_by_role('button', name='Zatwierdzam ten skrót').wait_for()
        from storage import open_db_readonly
        db = open_db_readonly(cfg.db_path)
        assert storage.records(db, 'character_short_examples')[0]['status']=='draft'
        db.close()
        frame.get_by_role('button', name='Zatwierdzam ten skrót').click()
        frame.locator('#reason').fill('Reviewed this exact short wording.')
        frame.get_by_role('button', name='Zapisz i przejdź dalej').click()
        frame.get_by_role('heading', name='Brak skrótów do zatwierdzenia.').wait_for()
        db=open_db_readonly(cfg.db_path)
        saved=storage.records(db, 'character_short_examples')[0]
        assert saved['status']=='approved'
        assert saved['body']['preferred']=='Check the actual result before claiming success.'
        assert len(storage.records(db, 'character_review_events'))==2
        db.close()
        assert errors==[]
        browser.close()


def test_browser_history_can_reach_the_third_page(server, env, monkeypatch):
    j,r,cfg,conn,pair=env
    for number in range(1,12):
        storage.insert_record(conn, 'character_pairs', dict(pair,id='history-'+str(number),
            status='rejected', created_at=f'2026-10-{number:02}T10:00:00Z',
            prompt_messages=[{'role':'user','content':f'History item {number}'}]))
    conn.commit()
    conn.close()
    monkeypatch.setattr(server.app, '_connect', type(server.app)._connect.__get__(server.app))
    with fixture_host(server) as url, playwright.sync_playwright() as driver:
        browser=driver.chromium.launch(headless=True)
        page=browser.new_page()
        page.goto(url)
        frame=page.frame_locator('iframe')
        frame.get_by_role('tab',name='Historia').click()
        frame.locator('#queue').get_by_role('button').filter(has_text='History item 11').wait_for()
        frame.get_by_role('button',name='Pokaż kolejne 5').click()
        frame.locator('#queue').get_by_role('button').filter(has_text='History item 6').wait_for()
        frame.get_by_role('button',name='Pokaż kolejne 5').click()
        frame.locator('#queue').get_by_role('button').filter(has_text='History item 1').wait_for()
        assert frame.locator('#queue').get_by_role('button').count()==1
        browser.close()


def test_browser_delete_without_answer_or_reason_stays_deleted_after_reload(server, env, monkeypatch):
    j, r, cfg, conn, pair = env
    conn.close()
    monkeypatch.setattr(server.app, '_connect', type(server.app)._connect.__get__(server.app))
    with fixture_host(server) as url, playwright.sync_playwright() as driver:
        browser = driver.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url)
        frame = page.frame_locator('iframe')
        remove = frame.get_by_role('button', name='Usuń parę na zawsze', exact=True)
        remove.wait_for()
        assert frame.locator('#reason').input_value() == ''
        assert not frame.locator('#answer-left').get_attribute('class').endswith('selected')
        remove.click()
        frame.get_by_role('heading', name='Nie ma teraz par do oceny.').wait_for()
        page.reload()
        frame.get_by_role('heading', name='Nie ma teraz par do oceny.').wait_for()
        frame.get_by_role('tab', name='Historia').click()
        frame.get_by_role('heading', name='Nie ma jeszcze decyzji.').wait_for()
        from storage import open_db_readonly
        with open_db_readonly(cfg.db_path) as db:
            assert storage.records(db, 'character_pairs') == []
            assert storage.records(db, 'character_review_events')[0]['action'] == 'delete'
        browser.close()


def test_browser_deletion_reason_survives_reload_and_is_saved(server, env, monkeypatch):
    j, r, cfg, conn, pair = env
    conn.close()
    monkeypatch.setattr(server.app, '_connect', type(server.app)._connect.__get__(server.app))
    with fixture_host(server) as url, playwright.sync_playwright() as driver:
        browser = driver.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url)
        frame = page.frame_locator('iframe')
        frame.get_by_text('Powód usunięcia (opcjonalnie)', exact=True).click()
        reason = 'Ten projekt jest zamknięty. Usuń jego plany z pamięci.'
        frame.locator('#deletion-reason').fill(reason)
        page.reload()
        frame.locator('#deletion-reason').wait_for(state='visible')
        assert frame.locator('#deletion-reason').input_value() == reason
        frame.get_by_role('button', name='Usuń parę na zawsze', exact=True).click()
        frame.get_by_role('heading', name='Nie ma teraz par do oceny.').wait_for()
        from storage import open_db_readonly
        with open_db_readonly(cfg.db_path) as db:
            event = storage.records(db, 'character_review_events')[0]
            assert event['reason'] == reason
            assert event['subject']['topic'] == 'Task'
            assert storage.records(db, 'character_pairs') == []
        assert page.frames[1].evaluate("Object.keys(localStorage).filter(k=>k.startsWith('dream-review:p:')).length") == 0
        browser.close()


def test_browser_displays_original_communication_feedback(server, env, monkeypatch):
    j, r, cfg, conn, pair = env
    feedback = 'Nie twierdź, że działa, zanim sprawdzisz.'
    moment = storage.insert_record(conn, 'character_moments', dict(id='feedback-source', revision=1,
        constitution_sha256='abc', user_correction=feedback))
    storage.update_record(conn, 'character_pairs', dict(pair, revision=2, moment_id=moment['id'],
        provenance={**pair['provenance'], 'source_content_sha256':moment['content_sha256']}),
        expected_sha=pair['content_sha256'])
    conn.commit()
    conn.close()
    monkeypatch.setattr(server.app, '_connect', type(server.app)._connect.__get__(server.app))
    with fixture_host(server) as url, playwright.sync_playwright() as driver:
        browser = driver.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url)
        frame = page.frame_locator('iframe')
        frame.get_by_role('heading', name='Twoja uwaga do poprzedniej odpowiedzi', exact=True).wait_for()
        assert frame.get_by_text(feedback, exact=True).count() == 1
        browser.close()


def test_delete_on_second_page_does_not_skip_next_unseen_item(server, env, monkeypatch):
    _, _, cfg, conn, base = env
    for n in range(1, 13):
        row = dict(base, id='p' if n == 1 else f'pair-{n}',
                   created_at=f'2026-10-{n:02}T10:00:00Z',
                   prompt_messages=[{'role':'user', 'content':f'Question {n}'}])
        if n == 1:
            storage.update_record(conn, 'character_pairs', dict(row, revision=2),
                                  expected_sha=base['content_sha256'])
        else:
            storage.insert_record(conn, 'character_pairs', row)
    conn.commit()
    conn.close()
    monkeypatch.setattr(server.app, '_connect', type(server.app)._connect.__get__(server.app))
    with fixture_host(server) as url, playwright.sync_playwright() as driver:
        browser = driver.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url)
        frame = page.frame_locator('iframe')
        frame.get_by_role('heading', name='Która odpowiedź lepiej Ci służy?').wait_for()
        frame.locator('#next-batch').click()
        frame.locator('#queue').get_by_role('button').filter(has_text='Question 6').wait_for()
        frame.get_by_role('button', name='Usuń parę na zawsze', exact=True).click()
        frame.locator('#queue').get_by_role('button').filter(has_text='Question 7').wait_for()
        frame.locator('#next-batch').click()
        frame.locator('#queue').get_by_role('button').filter(has_text='Question 12').wait_for()
        titles = frame.locator('#queue').get_by_role('button').all_text_contents()
        browser.close()
        assert any('Question 11' in title for title in titles), titles
