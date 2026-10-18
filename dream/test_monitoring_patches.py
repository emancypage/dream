import copy
import json
import unittest

import monitoring_dashboard
import monitoring_prometheus


PROMETHEUS_FIXTURE = """global:
  scrape_interval: 15s

scrape_configs:
  - job_name: personal-rag-workers
    static_configs:
      - targets: [\"192.168.1.22:30091\"]
"""


class PrometheusPatchTest(unittest.TestCase):
    def test_adds_dream_exporter_job_with_explicit_target(self):
        patched = monitoring_prometheus.patch_text(PROMETHEUS_FIXTURE, "192.168.1.18:9409")

        self.assertIn("  - job_name: dream\n", patched)
        self.assertIn('targets: ["192.168.1.18:9409"]', patched)
        self.assertIn("labels: {instance: dream}", patched)
        self.assertEqual(patched.count("job_name: dream"), 1)

    def test_reapplying_keeps_existing_job_and_does_not_guess_target(self):
        once = monitoring_prometheus.patch_text(PROMETHEUS_FIXTURE, "dream-host:9409")
        twice = monitoring_prometheus.patch_text(once, "another-host:9409")

        self.assertEqual(twice, once)

    def test_missing_anchor_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "personal-rag-workers"):
            monitoring_prometheus.patch_text("scrape_configs:\n", "dream-host:9409")

    def test_target_injection_is_rejected(self):
        with self.assertRaises(ValueError):
            monitoring_prometheus.patch_text(
                PROMETHEUS_FIXTURE,
                'dream-host:9409"\n  - job_name: bad',
            )


def dashboard_fixture():
    return {
        "title": "RPi Fleet",
        "uid": "rpi-fleet",
        "version": 7,
        "schemaVersion": 42,
        "panels": [
            {
                "id": 39,
                "type": "timeseries",
                "title": "Existing panel",
                "gridPos": {"x": 0, "y": 20, "w": 24, "h": 8},
            }
        ],
    }


class DashboardPatchTest(unittest.TestCase):
    def setUp(self):
        self.patched = monitoring_dashboard.patch_dashboard(dashboard_fixture())
        self.by_id = {panel["id"]: panel for panel in self.patched["panels"]}

    def test_adds_owned_health_panels_after_existing_content(self):
        self.assertEqual(self.by_id[40]["title"], "Dream — Stan nocnego przebiegu")
        self.assertEqual(self.by_id[41]["title"], "Dream — Wiek ostatniego przebiegu")
        self.assertEqual(self.by_id[42]["title"], "Dream — Etap i wynik")
        self.assertEqual(self.by_id[43]["title"], "Dream — Historia zdrowia")
        self.assertEqual(self.by_id[40]["gridPos"]["y"], 28)

    def test_health_stat_has_explicit_value_mappings_and_query(self):
        panel = self.by_id[40]

        self.assertEqual(
            panel["targets"][0]["expr"],
            'dream_run_health{job="dream",instance="dream"}',
        )
        mappings = panel["fieldConfig"]["defaults"]["mappings"][0]["options"]
        self.assertEqual(mappings["3"]["text"], "OK")
        self.assertEqual(mappings["0"]["text"], "ERROR")
        self.assertEqual(mappings["-1"]["text"], "BRAK DANYCH")

    def test_dashboard_queries_expose_age_stage_and_history(self):
        self.assertEqual(
            self.by_id[41]["targets"][0]["expr"],
            'time() - dream_last_run_timestamp_seconds{job="dream",instance="dream"}',
        )
        self.assertEqual(
            self.by_id[42]["targets"][0]["expr"],
            'dream_last_run_info{job="dream",instance="dream"} == 1',
        )
        self.assertEqual(
            self.by_id[43]["targets"][0]["expr"],
            'dream_run_health{job="dream",instance="dream"}',
        )

    def test_reapplying_replaces_only_owned_panels_and_does_not_mutate_input(self):
        source = dashboard_fixture()
        original = copy.deepcopy(source)
        first = monitoring_dashboard.patch_dashboard(source)
        first["panels"].append({"id": 99, "type": "stat", "title": "User panel"})

        second = monitoring_dashboard.patch_dashboard(first)

        self.assertEqual(source, original)
        self.assertEqual([panel["id"] for panel in second["panels"]].count(40), 1)
        self.assertIn(99, [panel["id"] for panel in second["panels"]])
        self.assertEqual(second["version"], first["version"] + 1)
        json.dumps(second)


if __name__ == "__main__":
    unittest.main()
