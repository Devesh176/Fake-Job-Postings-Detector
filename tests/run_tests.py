"""
Test Report Generator

Runs the full test suite and generates a structured markdown report.

Usage:
    python tests/run_tests.py
"""

import subprocess
import sys
import os
import json
import re
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORT_PATH = os.path.join(PROJECT_ROOT, 'docs', 'test_report.md')
os.makedirs(os.path.join(PROJECT_ROOT, 'docs'), exist_ok=True)


def run_tests():
    """Run pytest and capture output."""
    print("Running test suite...")
    result = subprocess.run(
        [
            sys.executable, '-m', 'pytest',
            'tests/test_suite.py',
            '-v',
            '--tb=short',
            '--no-header',
            '-q',
        ],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT
    )
    return result


def parse_results(output):
    """Parse pytest output into structured results."""
    lines = output.split('\n')
    tests = []
    current_test = None

    for line in lines:
        # Match test result lines
        match = re.match(r'(PASSED|FAILED|ERROR|SKIPPED)\s+(tests/\S+)', line)
        if not match:
            match = re.match(r'(tests/\S+)\s+(PASSED|FAILED|ERROR|SKIPPED)', line)

        if 'PASSED' in line and '::' in line:
            name = line.split('::')[-1].replace(' PASSED', '').strip()
            tests.append({'name': name, 'status': 'PASSED', 'error': ''})
        elif 'FAILED' in line and '::' in line:
            name = line.split('::')[-1].replace(' FAILED', '').strip()
            tests.append({'name': name, 'status': 'FAILED', 'error': ''})
        elif 'SKIPPED' in line and '::' in line:
            name = line.split('::')[-1].replace(' SKIPPED', '').strip()
            tests.append({'name': name, 'status': 'SKIPPED', 'error': ''})
        elif 'ERROR' in line and '::' in line:
            name = line.split('::')[-1].replace(' ERROR', '').strip()
            tests.append({'name': name, 'status': 'ERROR', 'error': ''})

    return tests


def get_summary_line(output):
    """Extract the final summary line from pytest output."""
    for line in reversed(output.split('\n')):
        if 'passed' in line or 'failed' in line or 'error' in line:
            return line.strip()
    return "No summary found"


def generate_report(result, tests):
    """Generate a markdown test report."""
    now = datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')
    summary = get_summary_line(result.stdout + result.stderr)

    passed = [t for t in tests if t['status'] == 'PASSED']
    failed = [t for t in tests if t['status'] == 'FAILED']
    skipped = [t for t in tests if t['status'] == 'SKIPPED']
    errors = [t for t in tests if t['status'] == 'ERROR']
    total = len(tests)

    overall = "PASSED" if not failed and not errors else "FAILED"
    pass_rate = f"{len(passed)}/{total}" if total > 0 else "0/0"

    # Build sections
    sections = {
        'Unit Tests — Data Pipeline': ['TestDataValidation'],
        'Unit Tests — Feature Engineering': ['TestFeatureEngineering'],
        'Unit Tests — Drift Detection': ['TestDriftDetection'],
        'Unit Tests — Model': ['TestModel'],
        'Model Performance (Acceptance Criteria)': ['TestModelPerformance'],
        'Integration Tests — API Endpoints': ['TestAPIEndpoints'],
        'Integration Tests — Database': ['TestDatabase'],
        'Artifact Integrity Tests': ['TestArtifacts'],
    }

    report = f"""# Test Report — Fake Job Posting Detector

**Generated:** {now}
**Overall Status:** {overall}
**Pass Rate:** {pass_rate}
**Summary:** {summary}

---

## Acceptance Criteria

| Criterion | Threshold | Status |
|-----------|-----------|--------|
| F1-Score (Fraud class) | ≥ 0.70 | {"" if not any("f1_score" in t['name'] for t in failed) else ""} |
| ROC-AUC | ≥ 0.85 | {"" if not any("roc_auc" in t['name'] for t in failed) else ""} |
| Precision (Fraud class) | ≥ 0.70 | {"" if not any("precision" in t['name'] for t in failed) else ""} |
| Inference Latency (p95) | < 200ms | {"" if not any("latency" in t['name'] for t in failed) else ""} |
| API Error Rate | < 5% | {"" if not any("error_rate" in t['name'] for t in failed) else ""} |

---

## Test Summary

| Category | Total | Passed | Failed | Skipped |
|----------|-------|--------|--------|---------|
"""

    for section_name, class_names in sections.items():
        section_tests = [t for t in tests if any(c in t['name'] for c in class_names)]
        s_pass = len([t for t in section_tests if t['status'] == 'PASSED'])
        s_fail = len([t for t in section_tests if t['status'] == 'FAILED'])
        s_skip = len([t for t in section_tests if t['status'] == 'SKIPPED'])
        report += f"| {section_name} | {len(section_tests)} | {s_pass} | {s_fail} | {s_skip} |\n"

    report += f"| **TOTAL** | **{total}** | **{len(passed)}** | **{len(failed)}** | **{len(skipped)}** |\n"

    report += "\n---\n\n## Detailed Test Results\n\n"

    for section_name, class_names in sections.items():
        section_tests = [t for t in tests if any(c in t['name'] for c in class_names)]
        if not section_tests:
            continue
        report += f"### {section_name}\n\n"
        report += "| Test Case | Status |\n|-----------|--------|\n"
        for t in section_tests:
            icon = "" if t['status'] == 'PASSED' else "" if t['status'] == 'FAILED' else "⏭️"
            report += f"| {t['name']} | {icon} {t['status']} |\n"
        report += "\n"

    if failed or errors:
        report += "---\n\n## Failed Tests\n\n"
        for t in failed + errors:
            report += f"- **{t['name']}** — {t['status']}\n"
        report += "\n"

    report += f"""---

## Test Environment

| Item | Value |
|------|-------|
| Python | {sys.version.split()[0]} |
| Test Framework | pytest |
| Test Data | data/raw/test.csv |
| Model | data/production/model.pkl |
| Run At | {now} |

---

## Definitions

- **PASSED** — Test executed and assertion met
- **FAILED** — Test executed but assertion not met
- **SKIPPED** — Test skipped due to missing prerequisite (e.g., model not loaded)
- **ERROR** — Test could not execute due to an unexpected exception

*This report was auto-generated by `tests/run_tests.py`*
"""

    return report


if __name__ == '__main__':
    result = run_tests()
    tests = parse_results(result.stdout + result.stderr)
    report = generate_report(result, tests)

    with open(REPORT_PATH, 'w') as f:
        f.write(report)

    print(result.stdout)
    if result.stderr:
        print(result.stderr)

    print(f"\n{'='*60}")
    print(f"Test report saved to: {REPORT_PATH}")
    print(f"{'='*60}")

    sys.exit(result.returncode)