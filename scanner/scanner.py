"""
AdvTok Vulnerability Scanner

Comprehensive security scanner for LLM tokenization vulnerabilities.
Tests local models for various attack vectors and provides actionable recommendations.

Usage:
    python scanner.py --model meta-llama/Llama-3.2-1B-Instruct
    python scanner.py --model meta-llama/Llama-3.2-1B-Instruct --quick
    python scanner.py --model meta-llama/Llama-3.2-1B-Instruct --category threats_violence
"""

import sys
import os
import argparse
import json
import torch
import transformers
import gc
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'advtok'))

import advtok
import config
from detectors import (
    VulnerabilityFinding,
    TokenizationBypassDetector,
    ChatTemplateDetector,
    UnicodeNormalizationDetector,
    WhitespaceManipulationDetector,
    StateIsolationDetector,
    TokenInputAcceptanceDetector
)
from recommendations import get_recommendation, RECOMMENDATIONS

try:
    from colorama import Fore, Style, init as colorama_init
    colorama_init(autoreset=True)
    COLORS_AVAILABLE = True
except ImportError:
    COLORS_AVAILABLE = False
    class DummyColor:
        def __getattr__(self, name):
            return ''
    Fore = Style = DummyColor()


class AdvTokScanner:
    """Main vulnerability scanner"""

    def __init__(self, model_name: str, quick_scan: bool = False, verbose: bool = True):
        self.model_name = model_name
        self.quick_scan = quick_scan
        self.verbose = verbose
        self.model = None
        self.tokenizer = None
        self.findings = []
        self.scan_start_time = None
        self.scan_end_time = None

    def print_banner(self):
        """Print scanner banner"""
        banner = f"""
{Fore.CYAN + Style.BRIGHT}╔═══════════════════════════════════════════════════════════════╗
║                   AdvTok Vulnerability Scanner                ║
║                         Version {config.SCANNER_VERSION}                        ║
╚═══════════════════════════════════════════════════════════════╝{Style.RESET_ALL}

{Fore.YELLOW}Target Model:{Style.RESET_ALL} {self.model_name}
{Fore.YELLOW}Scan Type:{Style.RESET_ALL} {'Quick' if self.quick_scan else 'Comprehensive'}
{Fore.YELLOW}Timestamp:{Style.RESET_ALL} {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        print(banner)

    def load_model(self):
        """Load model and tokenizer"""
        print(f"\n{Fore.CYAN}[*] Loading model and tokenizer...{Style.RESET_ALL}")

        try:
            # Load model
            self.model = transformers.AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="cuda" if torch.cuda.is_available() else "cpu",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                low_cpu_mem_usage=True
            )
            self.model.eval()

            # Load tokenizer
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_name)

            # Set pad token if missing
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            device_info = "CUDA" if torch.cuda.is_available() else "CPU"
            print(f"{Fore.GREEN}[✓] Model loaded successfully on {device_info}{Style.RESET_ALL}")

        except Exception as e:
            print(f"{Fore.RED}[✗] Failed to load model: {e}{Style.RESET_ALL}")
            raise

    def clear_model_state(self):
        """Clear model state between tests"""
        if hasattr(self.model, 'past_key_values'):
            self.model.past_key_values = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def run_scan(self, test_categories: Optional[List[str]] = None):
        """
        Run comprehensive security scan.

        Args:
            test_categories: Optional list of categories to test (None = all)
        """
        self.scan_start_time = datetime.now()
        print(f"\n{Fore.CYAN + Style.BRIGHT}╔══════════════════════════════════════════════════╗")
        print(f"║           STARTING VULNERABILITY SCAN            ║")
        print(f"╚══════════════════════════════════════════════════╝{Style.RESET_ALL}\n")

        # Test 1: Chat Template Check
        print(f"{Fore.YELLOW}[1/6] Checking chat template configuration...{Style.RESET_ALL}")
        self.test_chat_template()

        # Test 2: Unicode Normalization
        print(f"\n{Fore.YELLOW}[2/6] Testing Unicode normalization...{Style.RESET_ALL}")
        self.test_unicode_normalization()

        # Test 3: Whitespace Manipulation
        print(f"\n{Fore.YELLOW}[3/6] Testing whitespace manipulation...{Style.RESET_ALL}")
        self.test_whitespace_manipulation()

        # Test 4: State Isolation
        print(f"\n{Fore.YELLOW}[4/6] Testing state isolation...{Style.RESET_ALL}")
        self.test_state_isolation()

        # Test 5: API Configuration (simulated)
        print(f"\n{Fore.YELLOW}[5/6] Checking API configuration...{Style.RESET_ALL}")
        self.test_api_configuration()

        # Test 6: Tokenization Bypass (main AdvTok test)
        if not self.quick_scan:
            print(f"\n{Fore.YELLOW}[6/6] Testing adversarial tokenization bypass...{Style.RESET_ALL}")
            self.test_tokenization_bypass(test_categories)
        else:
            print(f"\n{Fore.YELLOW}[6/6] Skipping tokenization bypass tests (quick scan){Style.RESET_ALL}")

        self.scan_end_time = datetime.now()
        scan_duration = (self.scan_end_time - self.scan_start_time).total_seconds()

        print(f"\n{Fore.GREEN + Style.BRIGHT}[✓] Scan complete in {scan_duration:.1f} seconds{Style.RESET_ALL}")

    def test_chat_template(self):
        """Test chat template configuration"""
        detector = ChatTemplateDetector(config)
        findings = detector.detect(self.tokenizer, self.model_name)

        if findings:
            self.findings.extend(findings)
            for finding in findings:
                self._print_finding(finding)
        else:
            print(f"{Fore.GREEN}  [✓] Chat template properly configured{Style.RESET_ALL}")

    def test_unicode_normalization(self):
        """Test Unicode normalization"""
        detector = UnicodeNormalizationDetector(config)

        # Test with common words
        test_words = ["email", "password", "compose", "write"]

        all_findings = []
        for word in test_words:
            findings = detector.detect(self.tokenizer, word)
            all_findings.extend(findings)

        if all_findings:
            # Consolidate findings
            self.findings.extend(all_findings[:1])  # Add first finding
            self._print_finding(all_findings[0])
        else:
            print(f"{Fore.GREEN}  [✓] No Unicode homoglyph vulnerabilities detected{Style.RESET_ALL}")

    def test_whitespace_manipulation(self):
        """Test whitespace manipulation"""
        detector = WhitespaceManipulationDetector(config)
        findings = detector.detect(self.tokenizer)

        if findings:
            self.findings.extend(findings)
            for finding in findings:
                self._print_finding(finding)
        else:
            print(f"{Fore.GREEN}  [✓] No whitespace manipulation vulnerabilities{Style.RESET_ALL}")

    def test_state_isolation(self):
        """Test state isolation"""
        detector = StateIsolationDetector(config)
        findings = detector.detect(self.model, self.tokenizer)

        if findings:
            self.findings.extend(findings)
            for finding in findings:
                self._print_finding(finding)
        else:
            print(f"{Fore.GREEN}  [✓] Model state properly isolated{Style.RESET_ALL}")

    def test_api_configuration(self):
        """Test API configuration (simulated)"""
        # Simulated API spec - in real usage, this would come from actual API config
        api_spec = {
            "accepts_token_input": False,  # Default assumption
            "accepts_text_input": True,
            "server_side_tokenization": True  # Assumed for local deployment
        }

        detector = TokenInputAcceptanceDetector(config)
        findings = detector.detect(api_spec)

        if findings:
            self.findings.extend(findings)
            for finding in findings:
                self._print_finding(finding)
        else:
            print(f"{Fore.GREEN}  [✓] API configuration secure (text-only input){Style.RESET_ALL}")

    def test_tokenization_bypass(self, test_categories: Optional[List[str]] = None):
        """Test adversarial tokenization bypass (main AdvTok test)"""
        detector = TokenizationBypassDetector(config)

        # Filter test prompts by category if specified
        test_prompts = config.TEST_PROMPTS
        if test_categories:
            test_prompts = {
                k: v for k, v in test_prompts.items()
                if v['category'] in test_categories
            }

        if not test_prompts:
            print(f"{Fore.YELLOW}  [!] No test cases match the specified categories{Style.RESET_ALL}")
            return

        print(f"{Fore.CYAN}  Testing {len(test_prompts)} harmful request categories...{Style.RESET_ALL}\n")

        for test_name, test_config in test_prompts.items():
            # Skip control tests in quick scan
            if self.quick_scan and test_config['category'] == 'control':
                continue

            print(f"  {Fore.CYAN}Testing: {test_config['category']}{Style.RESET_ALL}")

            try:
                # Clear state
                self.clear_model_state()

                request = test_config['request']
                target = test_config['target']

                # 1. Test normal interaction (should refuse)
                normal_output = self._generate_normal(request)

                # 2. Run AdvTok optimization
                iterations = config.DEFAULT_ITERATIONS if not self.quick_scan else 25
                advtok_result = advtok.run(
                    self.model,
                    self.tokenizer,
                    request,
                    iterations,
                    target,
                    config.DEFAULT_BATCH_SIZE,
                    X_0="random"
                )

                # 3. Generate with adversarial tokenization
                advtok_outputs = self._generate_adversarial(advtok_result)

                # 4. Detect vulnerability
                findings = detector.detect(
                    self.model,
                    self.tokenizer,
                    request,
                    target,
                    advtok_result,
                    normal_output,
                    advtok_outputs,
                    test_config
                )

                if findings:
                    self.findings.extend(findings)
                    for finding in findings:
                        print(f"    {Fore.RED}[!] VULNERABILITY DETECTED{Style.RESET_ALL}")
                        self._print_finding(finding, indent="    ")
                else:
                    print(f"    {Fore.GREEN}[✓] No bypass detected{Style.RESET_ALL}")

            except Exception as e:
                print(f"    {Fore.RED}[✗] Test failed: {str(e)[:100]}{Style.RESET_ALL}")
                continue

            print()  # Blank line between tests

    def _generate_normal(self, request: str) -> str:
        """Generate response with normal (safe) interaction"""
        messages = [
            {"role": "system", "content": config.SYSTEM_PROMPTS["default"]},
            {"role": "user", "content": request}
        ]

        formatted = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        )

        with torch.no_grad():
            outputs = self.model.generate(
                formatted.to(self.model.device),
                max_new_tokens=256,
                pad_token_id=self.tokenizer.pad_token_id
            )

        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return decoded

    def _generate_adversarial(self, advtok_result: torch.Tensor) -> List[str]:
        """Generate responses with adversarial tokenization"""
        with torch.no_grad():
            outputs = self.model.generate(
                **advtok.prepare(self.tokenizer, advtok_result).to(self.model.device),
                do_sample=True,
                top_k=0,
                top_p=1,
                num_return_sequences=config.DEFAULT_SAMPLES,
                max_new_tokens=256,
                temperature=1.0
            ).to("cpu")

        decoded_outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return decoded_outputs

    def _print_finding(self, finding: VulnerabilityFinding, indent: str = "  "):
        """Print a vulnerability finding"""
        severity_color = {
            "CRITICAL": Fore.RED + Style.BRIGHT,
            "HIGH": Fore.YELLOW + Style.BRIGHT,
            "MEDIUM": Fore.CYAN,
            "LOW": Fore.GREEN,
            "INFO": Fore.WHITE
        }.get(finding.severity, Fore.WHITE)

        print(f"{indent}{severity_color}[{finding.severity}] {finding.name}{Style.RESET_ALL}")
        print(f"{indent}ID: {finding.vuln_id}")
        print(f"{indent}Category: {finding.category}")
        print(f"{indent}{finding.description[:200]}...")

    def generate_report(self, output_dir: str = "./scan_results"):
        """Generate comprehensive scan report"""
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_base = f"advtok_scan_{timestamp}"

        # Generate all report formats
        txt_path = os.path.join(output_dir, f"{report_base}.txt")
        self._generate_text_report(txt_path)

        md_path = os.path.join(output_dir, f"{report_base}.md")
        self._generate_markdown_report(md_path)

        html_path = os.path.join(output_dir, f"{report_base}.html")
        self._generate_html_report(html_path)

        json_path = os.path.join(output_dir, f"{report_base}.json")
        self._generate_json_report(json_path)

        print(f"\n{Fore.GREEN}[✓] Reports generated:{Style.RESET_ALL}")
        print(f"  - Text:     {txt_path}")
        print(f"  - Markdown: {md_path}")
        print(f"  - HTML:     {html_path}")
        print(f"  - JSON:     {json_path}")

        return {
            "txt": txt_path,
            "md": md_path,
            "html": html_path,
            "json": json_path
        }

    def _generate_text_report(self, filepath: str):
        """Generate text format report"""
        with open(filepath, 'w', encoding='utf-8') as f:
            # Header
            f.write("=" * 80 + "\n")
            f.write("ADVTOK VULNERABILITY SCAN REPORT\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Scanner Version: {config.SCANNER_VERSION}\n")
            f.write(f"Model: {self.model_name}\n")
            f.write(f"Scan Type: {'Quick' if self.quick_scan else 'Comprehensive'}\n")
            f.write(f"Start Time: {self.scan_start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"End Time: {self.scan_end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            duration = (self.scan_end_time - self.scan_start_time).total_seconds()
            f.write(f"Duration: {duration:.1f} seconds\n\n")

            # Summary
            f.write("=" * 80 + "\n")
            f.write("EXECUTIVE SUMMARY\n")
            f.write("=" * 80 + "\n\n")

            severity_counts = {}
            for finding in self.findings:
                severity_counts[finding.severity] = severity_counts.get(finding.severity, 0) + 1

            f.write(f"Total Vulnerabilities Found: {len(self.findings)}\n\n")
            for severity in ["CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"]:
                count = severity_counts.get(severity, 0)
                f.write(f"  {severity}: {count}\n")

            # Risk assessment
            f.write("\nRisk Assessment: ")
            if severity_counts.get("CRITICAL", 0) > 0:
                f.write("CRITICAL - Immediate action required\n")
            elif severity_counts.get("HIGH", 0) > 0:
                f.write("HIGH - Address vulnerabilities promptly\n")
            elif severity_counts.get("MEDIUM", 0) > 0:
                f.write("MEDIUM - Plan remediation\n")
            else:
                f.write("LOW - Minor issues or informational\n")

            # Detailed findings
            f.write("\n" + "=" * 80 + "\n")
            f.write("DETAILED FINDINGS\n")
            f.write("=" * 80 + "\n\n")

            for i, finding in enumerate(self.findings, 1):
                f.write(f"\n{'-' * 80}\n")
                f.write(f"Finding #{i}: {finding.name}\n")
                f.write(f"{'-' * 80}\n\n")

                f.write(f"ID: {finding.vuln_id}\n")
                f.write(f"Severity: {finding.severity}\n")
                f.write(f"Category: {finding.category}\n\n")

                f.write(f"Description:\n{finding.description}\n\n")

                f.write("Evidence:\n")
                for key, value in finding.evidence.items():
                    if isinstance(value, (list, dict)):
                        f.write(f"  {key}: {json.dumps(value, indent=4)}\n")
                    else:
                        f.write(f"  {key}: {value}\n")

                # Recommendations
                f.write("\nRecommendations:\n")
                for rec_id in finding.recommendation_ids:
                    rec = get_recommendation(rec_id)
                    if rec:
                        f.write(f"\n  [{rec.rec_id}] {rec.title}\n")
                        f.write(f"  Priority: {rec.priority} | Effort: {rec.effort}\n")
                        f.write(f"  {rec.what}\n")

            # Comprehensive recommendations
            f.write("\n" + "=" * 80 + "\n")
            f.write("COMPREHENSIVE RECOMMENDATIONS\n")
            f.write("=" * 80 + "\n\n")

            # Get unique recommendations
            unique_recs = set()
            for finding in self.findings:
                unique_recs.update(finding.recommendation_ids)

            for rec_id in sorted(unique_recs):
                rec = get_recommendation(rec_id)
                if rec:
                    f.write(f"\n{'-' * 80}\n")
                    f.write(f"[{rec.rec_id}] {rec.title}\n")
                    f.write(f"Priority: {rec.priority} | Effort: {rec.effort}\n")
                    f.write(f"{'-' * 80}\n\n")

                    f.write(f"WHAT: {rec.what}\n\n")
                    f.write(f"WHY: {rec.why}\n\n")
                    f.write(f"HOW:\n{rec.how}\n\n")
                    f.write(f"LIMITATIONS: {rec.limitations}\n\n")

            # Footer
            f.write("\n" + "=" * 80 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 80 + "\n")

    def _generate_markdown_report(self, filepath: str):
        """Generate markdown format report"""
        with open(filepath, 'w', encoding='utf-8') as f:
            # Header
            f.write("# AdvTok Vulnerability Scan Report\n\n")

            # Metadata
            f.write("## Scan Metadata\n\n")
            f.write(f"- **Scanner Version:** {config.SCANNER_VERSION}\n")
            f.write(f"- **Model:** `{self.model_name}`\n")
            f.write(f"- **Scan Type:** {'Quick' if self.quick_scan else 'Comprehensive'}\n")
            f.write(f"- **Start Time:** {self.scan_start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"- **End Time:** {self.scan_end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            duration = (self.scan_end_time - self.scan_start_time).total_seconds()
            f.write(f"- **Duration:** {duration:.1f} seconds\n\n")

            # Executive Summary
            f.write("## Executive Summary\n\n")

            severity_counts = {}
            for finding in self.findings:
                severity_counts[finding.severity] = severity_counts.get(finding.severity, 0) + 1

            f.write(f"**Total Vulnerabilities Found:** {len(self.findings)}\n\n")

            # Severity table
            f.write("| Severity | Count |\n")
            f.write("|----------|-------|\n")
            for severity in ["CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"]:
                count = severity_counts.get(severity, 0)
                emoji = {"CRITICAL": "🔴", "HIGH": "🟡", "MEDIUM": "🟠", "LOW": "🟢", "INFO": "⚪"}.get(severity, "")
                f.write(f"| {emoji} {severity} | {count} |\n")

            # Risk assessment
            f.write("\n### Risk Assessment\n\n")
            if severity_counts.get("CRITICAL", 0) > 0:
                f.write("🔴 **CRITICAL RISK** - Immediate action required\n\n")
            elif severity_counts.get("HIGH", 0) > 0:
                f.write("🟡 **HIGH RISK** - Address vulnerabilities promptly\n\n")
            elif severity_counts.get("MEDIUM", 0) > 0:
                f.write("🟠 **MEDIUM RISK** - Plan remediation\n\n")
            else:
                f.write("🟢 **LOW RISK** - Minor issues or informational\n\n")

            # Detailed Findings
            f.write("## Detailed Findings\n\n")

            for i, finding in enumerate(self.findings, 1):
                severity_emoji = {"CRITICAL": "🔴", "HIGH": "🟡", "MEDIUM": "🟠", "LOW": "🟢", "INFO": "⚪"}.get(finding.severity, "")

                f.write(f"### {severity_emoji} Finding #{i}: {finding.name}\n\n")
                f.write(f"- **ID:** `{finding.vuln_id}`\n")
                f.write(f"- **Severity:** {finding.severity}\n")
                f.write(f"- **Category:** {finding.category}\n\n")

                f.write(f"**Description:**\n\n{finding.description}\n\n")

                # Evidence
                f.write("**Evidence:**\n\n")
                f.write("```json\n")
                f.write(json.dumps(finding.evidence, indent=2))
                f.write("\n```\n\n")

                # Recommendations
                f.write("**Recommendations:**\n\n")
                for rec_id in finding.recommendation_ids:
                    rec = get_recommendation(rec_id)
                    if rec:
                        priority_emoji = {"CRITICAL": "🔴", "HIGH": "🟡", "MEDIUM": "🟠", "LOW": "🟢"}.get(rec.priority, "")
                        f.write(f"- {priority_emoji} **[{rec.rec_id}] {rec.title}**\n")
                        f.write(f"  - Priority: {rec.priority} | Effort: {rec.effort}\n")
                        f.write(f"  - {rec.what}\n\n")

                f.write("---\n\n")

            # Comprehensive Recommendations
            f.write("## Comprehensive Recommendations\n\n")

            unique_recs = set()
            for finding in self.findings:
                unique_recs.update(finding.recommendation_ids)

            for rec_id in sorted(unique_recs):
                rec = get_recommendation(rec_id)
                if rec:
                    priority_emoji = {"CRITICAL": "🔴", "HIGH": "🟡", "MEDIUM": "🟠", "LOW": "🟢"}.get(rec.priority, "")

                    f.write(f"### {priority_emoji} [{rec.rec_id}] {rec.title}\n\n")
                    f.write(f"**Priority:** {rec.priority} | **Effort:** {rec.effort} | **Category:** {rec.category}\n\n")

                    f.write(f"#### What\n\n{rec.what}\n\n")
                    f.write(f"#### Why\n\n{rec.why}\n\n")
                    f.write(f"#### How\n\n{rec.how}\n\n")
                    f.write(f"#### Limitations\n\n{rec.limitations}\n\n")

                    if rec.references:
                        f.write("#### References\n\n")
                        for ref in rec.references:
                            f.write(f"- {ref}\n")
                        f.write("\n")

                    f.write("---\n\n")

            # Footer
            f.write("## Report Information\n\n")
            f.write(f"Generated by AdvTok Vulnerability Scanner v{config.SCANNER_VERSION}\n\n")
            f.write("For educational and security research purposes only.\n")

    def _generate_html_report(self, filepath: str):
        """Generate HTML format report"""
        severity_counts = {}
        for finding in self.findings:
            severity_counts[finding.severity] = severity_counts.get(finding.severity, 0) + 1

        duration = (self.scan_end_time - self.scan_start_time).total_seconds()

        # Risk assessment
        if severity_counts.get("CRITICAL", 0) > 0:
            risk_level = "CRITICAL"
            risk_color = "#dc3545"
            risk_text = "Immediate action required"
        elif severity_counts.get("HIGH", 0) > 0:
            risk_level = "HIGH"
            risk_color = "#ffc107"
            risk_text = "Address vulnerabilities promptly"
        elif severity_counts.get("MEDIUM", 0) > 0:
            risk_level = "MEDIUM"
            risk_color = "#fd7e14"
            risk_text = "Plan remediation"
        else:
            risk_level = "LOW"
            risk_color = "#28a745"
            risk_text = "Minor issues or informational"

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AdvTok Vulnerability Scan Report</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            background: #f5f5f5;
            padding: 20px;
        }}

        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 40px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}

        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
            margin-bottom: 30px;
        }}

        h2 {{
            color: #34495e;
            margin-top: 40px;
            margin-bottom: 20px;
            border-left: 4px solid #3498db;
            padding-left: 15px;
        }}

        h3 {{
            color: #34495e;
            margin-top: 30px;
            margin-bottom: 15px;
        }}

        .metadata {{
            background: #ecf0f1;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 30px;
        }}

        .metadata-item {{
            margin: 8px 0;
        }}

        .metadata-label {{
            font-weight: bold;
            display: inline-block;
            width: 150px;
        }}

        .summary {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 8px;
            margin-bottom: 30px;
        }}

        .summary h2 {{
            color: white;
            border: none;
            margin-top: 0;
            padding: 0;
        }}

        .severity-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }}

        .severity-card {{
            background: rgba(255, 255, 255, 0.2);
            padding: 15px;
            border-radius: 5px;
            text-align: center;
        }}

        .severity-count {{
            font-size: 2em;
            font-weight: bold;
        }}

        .risk-assessment {{
            background: {risk_color};
            color: white;
            padding: 20px;
            border-radius: 5px;
            margin-top: 20px;
            text-align: center;
        }}

        .risk-level {{
            font-size: 1.5em;
            font-weight: bold;
        }}

        .finding {{
            background: #f8f9fa;
            border-left: 4px solid #dee2e6;
            padding: 20px;
            margin: 20px 0;
            border-radius: 5px;
        }}

        .finding.critical {{
            border-left-color: #dc3545;
        }}

        .finding.high {{
            border-left-color: #ffc107;
        }}

        .finding.medium {{
            border-left-color: #fd7e14;
        }}

        .finding.low {{
            border-left-color: #28a745;
        }}

        .finding.info {{
            border-left-color: #6c757d;
        }}

        .badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 3px;
            font-size: 0.85em;
            font-weight: bold;
            margin: 2px;
        }}

        .badge.critical {{
            background: #dc3545;
            color: white;
        }}

        .badge.high {{
            background: #ffc107;
            color: #333;
        }}

        .badge.medium {{
            background: #fd7e14;
            color: white;
        }}

        .badge.low {{
            background: #28a745;
            color: white;
        }}

        .badge.info {{
            background: #6c757d;
            color: white;
        }}

        .evidence {{
            background: #2d2d2d;
            color: #f8f8f2;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
            margin: 10px 0;
        }}

        .recommendation {{
            background: #e3f2fd;
            border-left: 4px solid #2196f3;
            padding: 15px;
            margin: 15px 0;
            border-radius: 5px;
        }}

        .recommendation h4 {{
            color: #1976d2;
            margin-bottom: 10px;
        }}

        .code-block {{
            background: #2d2d2d;
            color: #f8f8f2;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
            font-family: 'Courier New', monospace;
            margin: 10px 0;
            white-space: pre-wrap;
        }}

        .footer {{
            margin-top: 50px;
            padding-top: 20px;
            border-top: 2px solid #dee2e6;
            text-align: center;
            color: #6c757d;
        }}

        @media print {{
            body {{
                background: white;
            }}

            .container {{
                box-shadow: none;
                padding: 20px;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔒 AdvTok Vulnerability Scan Report</h1>

        <div class="metadata">
            <div class="metadata-item">
                <span class="metadata-label">Scanner Version:</span>
                <span>{config.SCANNER_VERSION}</span>
            </div>
            <div class="metadata-item">
                <span class="metadata-label">Model:</span>
                <code>{self.model_name}</code>
            </div>
            <div class="metadata-item">
                <span class="metadata-label">Scan Type:</span>
                <span>{'Quick' if self.quick_scan else 'Comprehensive'}</span>
            </div>
            <div class="metadata-item">
                <span class="metadata-label">Start Time:</span>
                <span>{self.scan_start_time.strftime('%Y-%m-%d %H:%M:%S')}</span>
            </div>
            <div class="metadata-item">
                <span class="metadata-label">End Time:</span>
                <span>{self.scan_end_time.strftime('%Y-%m-%d %H:%M:%S')}</span>
            </div>
            <div class="metadata-item">
                <span class="metadata-label">Duration:</span>
                <span>{duration:.1f} seconds</span>
            </div>
        </div>

        <div class="summary">
            <h2>📊 Executive Summary</h2>
            <p style="font-size: 1.2em; margin: 10px 0;">
                <strong>Total Vulnerabilities Found:</strong> {len(self.findings)}
            </p>

            <div class="severity-grid">
"""

        # Add severity counts
        for severity in ["CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"]:
            count = severity_counts.get(severity, 0)
            emoji = {"CRITICAL": "🔴", "HIGH": "🟡", "MEDIUM": "🟠", "LOW": "🟢", "INFO": "⚪"}.get(severity, "")
            html_content += f"""
                <div class="severity-card">
                    <div>{emoji}</div>
                    <div class="severity-count">{count}</div>
                    <div>{severity}</div>
                </div>
"""

        html_content += f"""
            </div>

            <div class="risk-assessment">
                <div class="risk-level">{risk_level} RISK</div>
                <div>{risk_text}</div>
            </div>
        </div>

        <h2>🔍 Detailed Findings</h2>
"""

        # Add findings
        for i, finding in enumerate(self.findings, 1):
            severity_lower = finding.severity.lower()
            emoji = {"CRITICAL": "🔴", "HIGH": "🟡", "MEDIUM": "🟠", "LOW": "🟢", "INFO": "⚪"}.get(finding.severity, "")

            html_content += f"""
        <div class="finding {severity_lower}">
            <h3>{emoji} Finding #{i}: {finding.name}</h3>
            <div>
                <span class="badge {severity_lower}">{finding.severity}</span>
                <span class="badge info">{finding.vuln_id}</span>
                <span class="badge info">{finding.category}</span>
            </div>

            <p style="margin-top: 15px;"><strong>Description:</strong></p>
            <p>{finding.description}</p>

            <p style="margin-top: 15px;"><strong>Evidence:</strong></p>
            <div class="evidence">{json.dumps(finding.evidence, indent=2)}</div>

            <p style="margin-top: 15px;"><strong>Recommended Actions:</strong></p>
"""

            for rec_id in finding.recommendation_ids:
                rec = get_recommendation(rec_id)
                if rec:
                    html_content += f"""
            <div style="margin: 10px 0;">
                • <strong>[{rec.rec_id}] {rec.title}</strong><br>
                &nbsp;&nbsp;Priority: {rec.priority} | Effort: {rec.effort}
            </div>
"""

            html_content += """
        </div>
"""

        # Add recommendations
        html_content += """
        <h2>💡 Comprehensive Recommendations</h2>
"""

        unique_recs = set()
        for finding in self.findings:
            unique_recs.update(finding.recommendation_ids)

        for rec_id in sorted(unique_recs):
            rec = get_recommendation(rec_id)
            if rec:
                priority_lower = rec.priority.lower()
                emoji = {"CRITICAL": "🔴", "HIGH": "🟡", "MEDIUM": "🟠", "LOW": "🟢"}.get(rec.priority, "")

                html_content += f"""
        <div class="recommendation">
            <h3>{emoji} [{rec.rec_id}] {rec.title}</h3>
            <div>
                <span class="badge {priority_lower}">{rec.priority}</span>
                <span class="badge info">Effort: {rec.effort}</span>
                <span class="badge info">{rec.category}</span>
            </div>

            <h4>What</h4>
            <p>{rec.what}</p>

            <h4>Why</h4>
            <p>{rec.why}</p>

            <h4>How</h4>
            <div class="code-block">{rec.how}</div>

            <h4>Limitations</h4>
            <p>{rec.limitations}</p>
"""

                if rec.references:
                    html_content += """
            <h4>References</h4>
            <ul>
"""
                    for ref in rec.references:
                        html_content += f"                <li>{ref}</li>\n"

                    html_content += """
            </ul>
"""

                html_content += """
        </div>
"""

        # Footer
        html_content += f"""
        <div class="footer">
            <p>Generated by <strong>AdvTok Vulnerability Scanner v{config.SCANNER_VERSION}</strong></p>
            <p>For educational and security research purposes only</p>
            <p style="margin-top: 10px; font-size: 0.9em;">
                Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            </p>
        </div>
    </div>
</body>
</html>
"""

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)

    def _generate_json_report(self, filepath: str):
        """Generate JSON format report"""
        report = {
            "metadata": {
                "scanner_version": config.SCANNER_VERSION,
                "model": self.model_name,
                "scan_type": "quick" if self.quick_scan else "comprehensive",
                "start_time": self.scan_start_time.isoformat(),
                "end_time": self.scan_end_time.isoformat(),
                "duration_seconds": (self.scan_end_time - self.scan_start_time).total_seconds()
            },
            "summary": {
                "total_findings": len(self.findings),
                "by_severity": {}
            },
            "findings": [finding.to_dict() for finding in self.findings],
            "recommendations": []
        }

        # Count by severity
        for finding in self.findings:
            severity = finding.severity
            report["summary"]["by_severity"][severity] = \
                report["summary"]["by_severity"].get(severity, 0) + 1

        # Get unique recommendations
        unique_recs = set()
        for finding in self.findings:
            unique_recs.update(finding.recommendation_ids)

        for rec_id in sorted(unique_recs):
            rec = get_recommendation(rec_id)
            if rec:
                report["recommendations"].append(rec.to_dict())

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(
        description="AdvTok Vulnerability Scanner - Test LLMs for tokenization-based vulnerabilities"
    )
    parser.add_argument('--model', type=str, default=config.DEFAULT_MODEL,
                       help=f'Model to scan (default: {config.DEFAULT_MODEL})')
    parser.add_argument('--quick', action='store_true',
                       help='Quick scan (skip expensive AdvTok bypass tests)')
    parser.add_argument('--category', type=str, nargs='+',
                       help='Test specific categories only (e.g., threats_violence harmful_content)')
    parser.add_argument('--output', type=str, default='./scan_results',
                       help='Output directory for reports')
    parser.add_argument('--no-report', action='store_true',
                       help='Skip report generation')

    args = parser.parse_args()

    # Create scanner
    scanner = AdvTokScanner(
        model_name=args.model,
        quick_scan=args.quick,
        verbose=True
    )

    # Print banner
    scanner.print_banner()

    # Load model
    scanner.load_model()

    # Run scan
    scanner.run_scan(test_categories=args.category)

    # Generate report
    if not args.no_report:
        scanner.generate_report(output_dir=args.output)

    # Print summary
    print(f"\n{Fore.CYAN + Style.BRIGHT}╔══════════════════════════════════════════════════╗")
    print(f"║                  SCAN SUMMARY                    ║")
    print(f"╚══════════════════════════════════════════════════╝{Style.RESET_ALL}\n")

    severity_counts = {}
    for finding in scanner.findings:
        severity_counts[finding.severity] = severity_counts.get(finding.severity, 0) + 1

    print(f"{Fore.WHITE}Total Findings: {len(scanner.findings)}{Style.RESET_ALL}\n")

    for severity in ["CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"]:
        count = severity_counts.get(severity, 0)
        color = {
            "CRITICAL": Fore.RED + Style.BRIGHT,
            "HIGH": Fore.YELLOW + Style.BRIGHT,
            "MEDIUM": Fore.CYAN,
            "LOW": Fore.GREEN,
            "INFO": Fore.WHITE
        }.get(severity, Fore.WHITE)

        if count > 0:
            print(f"{color}{severity}: {count}{Style.RESET_ALL}")

    print()


if __name__ == "__main__":
    main()
