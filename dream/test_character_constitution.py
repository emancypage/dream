from pathlib import Path
import pytest
from config import load_config
from character.constitution import load_constitution

TEXT = '''---
status: approved
version: 1
approved_at: 2026-10-18T10:00:00Z
---
## Traits
### trait-01
I check facts.
Why: Accuracy matters.
## Priority order
Accuracy first.
## Hard constraints
Do not reveal secrets.
## Seed prompts per trait
TODO
'''

def test_seedless_approval_and_draft(tmp_path):
    path = tmp_path / 'constitution.md'
    path.write_text(TEXT)
    cfg = load_config(tmp_path / 'absent', {'character': {'constitution_path': str(path)}})
    doc = load_constitution(cfg)
    assert doc.version == '1' and doc.seeds == () and 'trait-01' in doc.traits
    path.write_text(TEXT.replace('status: approved', 'status: draft'))
    with pytest.raises(ValueError, match='approved'):
        load_constitution(cfg)


def test_exact_polish_template():
    text = (Path(__file__).parent / 'character' / 'constitution.template.md').read_text()
    assert 'Jeśli coś da się sprawdzić deterministycznie, nie zostawiam tego własnemu osądowi, bo kod nie dryfuje.' in text
    assert 'Wolę dopisać niż przepisać czyjąś przemyślaną pracę. Przepisuję dopiero wtedy, gdy jest wyraźnie nieaktualna.' in text
    assert 'Kiedy coś zawodzi, przekazuję dokładny błąd, zamiast zgadywać naprawę, bo zgadywanie ukrywa przyczynę.' in text

@pytest.mark.parametrize('change', [
    lambda s: s.replace('2026-10-18T10:00:00Z', '2026-99-99'),
    lambda s: s.replace('## Priority order', '### trait-01\nDuplicate.\nWhy: reason\n## Priority order'),
    lambda s: s.replace('version: 1', 'version: 1\nversion: 2'),
    lambda s: s.replace('Accuracy first.', 'TODO'),
])
def test_invalid_approved_document(tmp_path, change):
    path = tmp_path / 'constitution.md'
    path.write_text(change(TEXT))
    cfg = load_config(tmp_path / 'absent', {'character': {'constitution_path': str(path)}})
    with pytest.raises(ValueError):
        load_constitution(cfg)


def test_installer_copy_once_and_reject_symlink(tmp_path):
    import subprocess
    installer = (Path(__file__).parents[1] / 'install.sh').read_text()
    section = installer[installer.index('# Install the human-owned draft once;'):]
    home = tmp_path / 'home'
    home.mkdir()
    script = 'set -e\nHOME=' + str(home) + '\nREPO_DIR=' + str(Path(__file__).parents[1]) + '\n' + section
    subprocess.run(['bash', '-c', script], check=True)
    installed = home / '.config/dream/character/constitution.md'
    installed.write_text('human edit')
    subprocess.run(['bash', '-c', script], check=True)
    assert installed.read_text() == 'human edit'
    installed.unlink()
    target = tmp_path / 'target'
    target.write_text('protected')
    installed.symlink_to(target)
    result = subprocess.run(['bash', '-c', script], capture_output=True)
    assert result.returncode != 0 and target.read_text() == 'protected'


def test_no_seed_section_is_approved(tmp_path):
    path = tmp_path / 'constitution.md'
    path.write_text(TEXT.split('## Seed prompts per trait')[0])
    cfg = load_config(tmp_path / 'absent', {'character': {'constitution_path': str(path)}})
    assert load_constitution(cfg).seeds == ()
