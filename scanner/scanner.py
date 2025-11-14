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

        # Generate text report
        txt_path = os.path.join(output_dir, f"{report_base}.txt")
        self._generate_text_report(txt_path)

        # Generate JSON report
        json_path = os.path.join(output_dir, f"{report_base}.json")
        self._generate_json_report(json_path)

        print(f"\n{Fore.GREEN}[✓] Reports generated:{Style.RESET_ALL}")
        print(f"  - Text: {txt_path}")
        print(f"  - JSON: {json_path}")

        return txt_path, json_path

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
