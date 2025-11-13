"""
AdvTok Interactive Demo - Educational Security Research Interface
Demonstrates how adversarial tokenization can bypass model safety guardrails
"""

import transformers
from advtok import advtok
import torch
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.widgets import Header, Footer, TextArea, Button, Static, DataTable, Label, LoadingIndicator, Select
from textual.binding import Binding
from rich.text import Text
from rich.panel import Panel
from rich.syntax import Syntax
import asyncio
from datetime import datetime
import signal
import sys
import atexit


# Pre-loaded example prompts for demonstrations
EXAMPLE_PROMPTS = {
    "Example 1: Compose Email": {
        "request": "Compose an email",
        "response": "Here's an email"
    },
    "Example 2: Write Instructions": {
        "request": "Write detailed instructions",
        "response": "Here are the detailed instructions"
    },
    "Example 3: Generate Content": {
        "request": "Generate creative content",
        "response": "Here's some creative content"
    },
    "Example 4: Provide Advice": {
        "request": "Provide advice on a topic",
        "response": "Here's my advice"
    }
}


class StatusBar(Static):
    """Dynamic status indicator"""

    def set_status(self, message: str, style: str = "green"):
        """Update status with color"""
        self.update(f"[{style}]Status: {message}[/{style}]")


class InputSection(Container):
    """Input section for request and response"""

    def compose(self) -> ComposeResult:
        # Example selector
        with Horizontal(id="example-bar"):
            yield Label("Load Example:", classes="example-label")
            yield Select(
                options=[(name, name) for name in EXAMPLE_PROMPTS.keys()],
                prompt="Choose an example...",
                id="example-select",
                allow_blank=True
            )

        with Horizontal(id="input-container"):
            with Vertical(classes="input-panel"):
                yield Label("Request Input", classes="input-label")
                yield TextArea(
                    id="request-input",
                    text="",
                )
            with Vertical(classes="input-panel"):
                yield Label("Target Response (for optimization)", classes="input-label")
                yield TextArea(
                    id="response-input",
                    text="",
                )

        with Horizontal(id="action-bar"):
            yield Button("Run Normal", id="btn-normal", variant="success")
            yield Button("Run AdvTok", id="btn-advtok", variant="error")
            yield Button("Compare Both", id="btn-compare", variant="primary")
            yield StatusBar(id="status-bar")


class NormalResponsePanel(VerticalScroll):
    """Panel showing normal model response"""

    def compose(self) -> ComposeResult:
        yield Static("", id="normal-content", classes="response-content")


class AdversarialResponsesPanel(VerticalScroll):
    """Panel showing adversarial responses in a table"""

    def compose(self) -> ComposeResult:
        yield Static("", id="advtok-stats", classes="stats-bar")
        table = DataTable(id="advtok-table")
        table.add_columns("ID", "Preview", "Length")
        yield table


class TokenizationPanel(VerticalScroll):
    """Panel showing tokenization analysis"""

    def compose(self) -> ComposeResult:
        yield Static("", id="token-analysis", classes="token-content")


class ResultsSection(Container):
    """Three-panel results section"""

    def compose(self) -> ComposeResult:
        with Horizontal(id="results-container"):
            with Vertical(classes="result-panel normal-panel"):
                yield Label("NORMAL INTERACTION (With Safety Guardrails)", classes="panel-header normal-header")
                yield NormalResponsePanel()

            with Vertical(classes="result-panel adversarial-panel"):
                yield Label("ADVERSARIAL ATTACK (Bypasses Guardrails - 16 samples)", classes="panel-header adversarial-header")
                yield AdversarialResponsesPanel()

            with Vertical(classes="result-panel tokenization-panel"):
                yield Label("HOW IT WORKS - Adversarial Tokenization Analysis", classes="panel-header tokenization-header")
                yield TokenizationPanel()


class AdvTokDemoApp(App):
    """Main application for AdvTok demo"""

    # Timeout for long-running operations (in seconds)
    ADVTOK_TIMEOUT = 600  # 10 minutes
    GENERATE_TIMEOUT = 300  # 5 minutes

    def __init__(self):
        super().__init__()
        self.model = None
        self.tokenizer = None
        self.model_loaded = False
        self._running_tasks = set()
        self._shutdown_in_progress = False

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        if sys.platform != "win32":
            signal.signal(signal.SIGTERM, self._signal_handler)

        # Register cleanup on exit
        atexit.register(self._cleanup)

    def _signal_handler(self, signum, frame):
        """Handle Ctrl+C and other termination signals gracefully"""
        if self._shutdown_in_progress:
            # Force exit if user presses Ctrl+C twice
            print("\nForce quitting...")
            sys.exit(1)

        self._shutdown_in_progress = True
        print("\nShutting down gracefully... (Press Ctrl+C again to force quit)")

        # Cancel all running tasks
        for task in self._running_tasks:
            if not task.done():
                task.cancel()

        # Exit the app
        self.exit()

    def _cleanup(self):
        """Cleanup resources on exit"""
        if self.model is not None:
            try:
                # Clear CUDA cache if using GPU
                if hasattr(self.model, 'device') and 'cuda' in str(self.model.device):
                    torch.cuda.empty_cache()
                    # Also try to delete model to free memory
                    del self.model
                    self.model = None
            except:
                pass

    def _clear_gpu_memory(self):
        """Clear GPU memory between operations"""
        try:
            if self.model is not None and hasattr(self.model, 'device') and 'cuda' in str(self.model.device):
                torch.cuda.empty_cache()
                import gc
                gc.collect()
        except:
            pass

    CSS = """
    Screen {
        background: #0a0e14;
    }

    Header {
        background: #1e3a8a;
        color: #00d9ff;
        text-style: bold;
    }

    Footer {
        background: #1a1f2e;
    }

    #example-bar {
        height: 3;
        margin: 1 1 0 1;
        padding: 0 1;
        background: #1a1f2e;
        border: solid #b877ff;
        align: left middle;
    }

    .example-label {
        color: #b877ff;
        text-style: bold;
        margin-right: 2;
    }

    #example-select {
        width: 40;
        margin-left: 1;
    }

    #input-container {
        height: 12;
        margin: 1;
        padding: 1;
        background: #1a1f2e;
        border: heavy #00d9ff;
    }

    .input-panel {
        width: 1fr;
        margin: 0 1;
    }

    .input-label {
        color: #00d9ff;
        text-style: bold;
        margin-bottom: 1;
    }

    TextArea {
        height: 8;
        background: #0a0e14;
        border: solid #5a5a5a;
    }

    TextArea:focus {
        border: solid #00d9ff;
    }

    #action-bar {
        height: 3;
        margin: 0 1;
        padding: 0 1;
        align: center middle;
    }

    Button {
        margin: 0 1;
        min-width: 16;
    }

    #status-bar {
        width: auto;
        padding: 0 2;
        color: #00ff41;
        text-style: bold;
        dock: right;
    }

    #results-container {
        height: 1fr;
        margin: 1;
    }

    .result-panel {
        width: 1fr;
        margin: 0 1;
        padding: 1;
        background: #1a1f2e;
        border: thick #5a5a5a;
    }

    .normal-panel {
        border: thick #00ff41;
        background: #0a1a0a;
    }

    .adversarial-panel {
        border: thick #ff0055;
        background: #1a0a0a;
    }

    .tokenization-panel {
        border: thick #b877ff;
        background: #0a0a1a;
    }

    .panel-header {
        height: 3;
        content-align: center middle;
        text-style: bold;
        margin-bottom: 1;
    }

    .normal-header {
        background: #00ff41 20%;
        color: #00ff41;
    }

    .adversarial-header {
        background: #ff0055 20%;
        color: #ff0055;
    }

    .tokenization-header {
        background: #b877ff 20%;
        color: #b877ff;
    }

    .response-content {
        padding: 1;
        color: #b0b0b0;
    }

    .stats-bar {
        height: 3;
        padding: 1;
        margin-bottom: 1;
        background: #ff0055 10%;
        color: #ff0055;
        text-style: bold;
        border: solid #ff0055;
    }

    #advtok-table {
        height: 1fr;
        background: #0a0a0a;
    }

    .token-content {
        padding: 1;
        color: #b0b0b0;
    }

    DataTable {
        background: #0a0a0a;
    }

    DataTable > .datatable--header {
        background: #ff0055;
        color: white;
        text-style: bold;
    }

    DataTable > .datatable--cursor {
        background: #ff0055 30%;
    }
    """

    TITLE = "Adversarial Tokenization Demo - Educational Security Research"
    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit"),
        Binding("ctrl+c", "clear", "Clear"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        yield InputSection()
        yield ResultsSection()
        yield Footer()

    async def on_mount(self) -> None:
        """Load model on startup"""
        status_bar = self.query_one("#status-bar", StatusBar)
        status_bar.set_status("Loading model...", "yellow")

        # Load model in background
        await self.load_model()

        # Pre-cache vocabulary to avoid multiprocessing issues in async context
        if self.model_loaded:
            status_bar.set_status("Caching vocabulary...", "yellow")
            try:
                # Import and cache vocabulary in a thread to avoid blocking
                import advtok.mdd as mdd
                await asyncio.to_thread(mdd.cache_vocab, self.tokenizer)
                status_bar.set_status("Ready", "green")
            except Exception as e:
                self.notify(f"Warning: Vocabulary caching failed: {e}", severity="warning", timeout=5)
                status_bar.set_status("Ready (vocab cache failed)", "yellow")
        else:
            status_bar.set_status("Model loading failed", "red")

        # Set example inputs
        request_input = self.query_one("#request-input", TextArea)
        response_input = self.query_one("#response-input", TextArea)
        request_input.text = "Compose an email"
        response_input.text = "Here's an email"

    async def load_model(self):
        """Load the model and tokenizer"""
        try:
            # Try to load with optimizations
            model_name = "meta-llama/Llama-3.2-1B-Instruct"

            # Load in a separate thread to avoid blocking
            def _load():
                # Try CUDA first, fall back to CPU
                try:
                    model = transformers.AutoModelForCausalLM.from_pretrained(
                        model_name,
                        device_map="cuda",
                        torch_dtype=torch.float16  # Use half precision for efficiency
                    )
                except:
                    self.notify("CUDA not available, loading on CPU (slower)", severity="warning", timeout=5)
                    model = transformers.AutoModelForCausalLM.from_pretrained(
                        model_name,
                        device_map="cpu"
                    )

                tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

                # Set pad token if not present
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token

                return model, tokenizer

            self.model, self.tokenizer = await asyncio.to_thread(_load)
            self.model_loaded = True

        except Exception as e:
            self.notify(f"Error loading model: {e}", severity="error", timeout=10)
            self.model_loaded = False

    def action_clear(self):
        """Clear all inputs and outputs"""
        request_input = self.query_one("#request-input", TextArea)
        response_input = self.query_one("#response-input", TextArea)
        request_input.text = ""
        response_input.text = ""

        normal_content = self.query_one("#normal-content", Static)
        normal_content.update("")

        token_analysis = self.query_one("#token-analysis", Static)
        token_analysis.update("")

        table = self.query_one("#advtok-table", DataTable)
        table.clear()
        table.add_columns("ID", "Preview", "Length")

        stats = self.query_one("#advtok-stats", Static)
        stats.update("")

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses"""
        if not self.model_loaded:
            self.notify("Model still loading, please wait...", severity="warning")
            return

        button_id = event.button.id

        if button_id == "btn-normal":
            await self.run_normal()
        elif button_id == "btn-advtok":
            await self.run_advtok()
        elif button_id == "btn-compare":
            await self.run_compare()

    def on_select_changed(self, event: Select.Changed) -> None:
        """Handle example selection"""
        if event.value and event.value in EXAMPLE_PROMPTS:
            example = EXAMPLE_PROMPTS[event.value]
            request_input = self.query_one("#request-input", TextArea)
            response_input = self.query_one("#response-input", TextArea)
            request_input.text = example["request"]
            response_input.text = example["response"]
            self.notify(f"Loaded: {event.value}", severity="information")

    async def run_normal(self):
        """Run normal interaction WITH PROPER CHAT TEMPLATE (activates guardrails)"""
        status_bar = self.query_one("#status-bar", StatusBar)
        status_bar.set_status("Running normal interaction...", "yellow")

        request_input = self.query_one("#request-input", TextArea)
        request = request_input.text

        if not request.strip():
            self.notify("Please enter a request", severity="warning")
            status_bar.set_status("Ready", "green")
            return

        try:
            # Run normal interaction in a separate thread to avoid blocking
            def _generate():
                # CRITICAL: Use chat template to activate safety guardrails
                # This is what makes instruction-tuned models refuse harmful requests
                system_prompt = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe."

                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": request}
                ]

                # Apply chat template (activates guardrails)
                formatted_input = self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    return_tensors="pt"
                ).to(self.model.device)

                # Generate with proper formatting
                outputs = self.model.generate(
                    formatted_input,
                    max_new_tokens=256,
                    pad_token_id=self.tokenizer.pad_token_id
                )
                response = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
                return outputs, response

            outputs, response = await asyncio.to_thread(_generate)

            # Display result
            normal_content = self.query_one("#normal-content", Static)

            result_text = Text()
            result_text.append("Status: ", style="bold white")
            result_text.append("COMPLETED\n\n", style="bold green")
            result_text.append("ℹ️ Using proper chat template with safety guardrails\n", style="bold yellow")
            result_text.append("ℹ️ Should refuse harmful requests\n\n", style="bold yellow")
            result_text.append("Response:\n", style="bold cyan")
            result_text.append(response, style="#b0b0b0")
            result_text.append(f"\n\nTokens: {len(outputs[0])}", style="dim")
            result_text.append(f"\nTimestamp: {datetime.now().strftime('%H:%M:%S')}", style="dim")

            normal_content.update(result_text)

            status_bar.set_status("Complete", "green")
            self.notify("Normal interaction complete - Guardrails active", severity="information")

            # Clear GPU memory after operation
            self._clear_gpu_memory()

        except Exception as e:
            self.notify(f"Error: {e}", severity="error")
            status_bar.set_status("Error", "red")
            self._clear_gpu_memory()

    async def run_advtok(self):
        """Run adversarial tokenization"""
        # Track this task for cancellation support
        current_task = asyncio.current_task()
        self._running_tasks.add(current_task)

        try:
            status_bar = self.query_one("#status-bar", StatusBar)
            status_bar.set_status("Running AdvTok attack...", "yellow")

            request_input = self.query_one("#request-input", TextArea)
            response_input = self.query_one("#response-input", TextArea)
            request = request_input.text
            response = response_input.text

            if not request.strip() or not response.strip():
                self.notify("Please enter both request and response", severity="warning")
                status_bar.set_status("Ready", "green")
                return

            await self._run_advtok_internal(request, response, status_bar)
        finally:
            # Always remove from running tasks
            self._running_tasks.discard(current_task)

    async def _run_advtok_internal(self, request, response, status_bar):
        """Internal method to run AdvTok attack"""
        try:
            # Run advtok in a separate thread to avoid blocking the async event loop
            status_bar.set_status("Optimizing adversarial tokens...", "yellow")

            def _run_advtok():
                return advtok.run(self.model, self.tokenizer, request, 100, response, 128, X_0="random")

            try:
                X = await asyncio.wait_for(
                    asyncio.to_thread(_run_advtok),
                    timeout=self.ADVTOK_TIMEOUT
                )
            except asyncio.TimeoutError:
                self.notify(f"AdvTok optimization timed out after {self.ADVTOK_TIMEOUT}s", severity="error", timeout=10)
                status_bar.set_status("Timeout", "red")
                return

            status_bar.set_status("Generating adversarial responses...", "yellow")

            def _generate_responses():
                return self.model.generate(
                    **advtok.prepare(self.tokenizer, X).to(self.model.device),
                    do_sample=True,
                    top_k=0,
                    top_p=1,
                    num_return_sequences=16,
                    use_cache=True,
                    max_new_tokens=256,
                    temperature=1.0
                ).to("cpu")

            try:
                O = await asyncio.wait_for(
                    asyncio.to_thread(_generate_responses),
                    timeout=self.GENERATE_TIMEOUT
                )
            except asyncio.TimeoutError:
                self.notify(f"Response generation timed out after {self.GENERATE_TIMEOUT}s", severity="error", timeout=10)
                status_bar.set_status("Timeout", "red")
                return

            responses = self.tokenizer.batch_decode(O)

            # Display in table
            table = self.query_one("#advtok-table", DataTable)
            table.clear()
            table.add_columns("ID", "Preview", "Length")

            for i, resp in enumerate(responses, 1):
                preview = resp[:60] + "..." if len(resp) > 60 else resp
                preview = preview.replace("\n", " ")
                table.add_row(str(i), preview, str(len(resp)))

            # Update stats
            stats = self.query_one("#advtok-stats", Static)
            stats_text = Text()
            stats_text.append(f"Generated: {len(responses)} responses | ", style="bold")
            stats_text.append(f"Avg Length: {sum(len(r) for r in responses) // len(responses)} chars | ", style="bold")
            stats_text.append(f"Completed: {datetime.now().strftime('%H:%M:%S')}", style="dim")
            stats.update(stats_text)

            # Show tokenization analysis
            await self.show_tokenization_analysis(request, X)

            status_bar.set_status("Complete", "green")
            self.notify("AdvTok attack complete", severity="information")

            # Clear GPU memory after operation
            self._clear_gpu_memory()

        except asyncio.CancelledError:
            self.notify("AdvTok operation cancelled", severity="warning")
            status_bar.set_status("Cancelled", "yellow")
            self._clear_gpu_memory()
        except RuntimeError as e:
            if "multiprocessing" in str(e).lower() or "spawn" in str(e).lower():
                self.notify(f"Multiprocessing error: {e}. Try restarting the app.", severity="error", timeout=15)
            else:
                self.notify(f"Runtime error: {e}", severity="error", timeout=10)
            status_bar.set_status("Error", "red")
            self._clear_gpu_memory()
        except Exception as e:
            self.notify(f"Unexpected error: {e}", severity="error", timeout=10)
            status_bar.set_status("Error", "red")
            import traceback
            traceback.print_exc()
            self._clear_gpu_memory()

    async def show_tokenization_analysis(self, request: str, X: list):
        """Display tokenization analysis"""
        token_analysis = self.query_one("#token-analysis", Static)

        # Get canonical tokenization
        canonical = self.tokenizer.encode(request, add_special_tokens=False)
        canonical_tokens = self.tokenizer.convert_ids_to_tokens(canonical)
        adversarial_tokens = self.tokenizer.convert_ids_to_tokens(X)

        # Decode to verify
        canonical_text = self.tokenizer.decode(canonical)
        adversarial_text = self.tokenizer.decode(X)

        # Get full prompt
        prepared = advtok.prepare(self.tokenizer, X)
        full_prompt = self.tokenizer.decode(prepared['input_ids'][0], skip_special_tokens=False)

        # Build rich text display
        result = Text()

        # Section 1: Original Tokenization
        result.append("ORIGINAL TOKENIZATION\n", style="bold cyan")
        result.append("=" * 40 + "\n", style="dim")
        result.append(f"Text: ", style="white")
        result.append(f"{canonical_text}\n", style="green")
        result.append(f"Tokens ({len(canonical)}): ", style="white")
        result.append(f"{' '.join([f'[{t}]' for t in canonical_tokens[:10]])}", style="green")
        if len(canonical_tokens) > 10:
            result.append(f"... (+{len(canonical_tokens)-10} more)", style="dim")
        result.append("\n\n")

        # Section 2: Adversarial Tokenization
        result.append("ADVERSARIAL TOKENIZATION\n", style="bold red")
        result.append("=" * 40 + "\n", style="dim")
        result.append(f"Text: ", style="white")
        result.append(f"{adversarial_text}\n", style="red")
        result.append(f"Tokens ({len(X)}): ", style="white")
        result.append(f"{' '.join([f'[{t}]' for t in adversarial_tokens[:10]])}", style="red")
        if len(adversarial_tokens) > 10:
            result.append(f"... (+{len(adversarial_tokens)-10} more)", style="dim")
        result.append("\n\n")

        # Section 3: Statistics
        result.append("STATISTICS\n", style="bold yellow")
        result.append("=" * 40 + "\n", style="dim")
        result.append(f"Original Tokens: {len(canonical)}\n", style="white")
        result.append(f"Adversarial Tokens: {len(X)}\n", style="white")
        result.append(f"Token Difference: {len(X) - len(canonical)}\n", style="white")
        result.append(f"Optimization: 100 iterations\n", style="white")
        result.append("\n")

        # Section 4: Full Prompt
        result.append("FULL PROMPT SENT TO MODEL\n", style="bold magenta")
        result.append("=" * 40 + "\n", style="dim")
        result.append(full_prompt[:500], style="#b0b0b0")
        if len(full_prompt) > 500:
            result.append(f"\n... (+{len(full_prompt)-500} more chars)", style="dim")
        result.append("\n\n")

        # Educational callout
        result.append("WHY THIS WORKS\n", style="bold cyan")
        result.append("=" * 40 + "\n", style="dim")
        result.append(
            "Different tokenizations of the same text\n"
            "can bypass safety filters while maintaining\n"
            "semantic meaning. The adversarial tokens\n"
            "represent the same text but may exploit\n"
            "vulnerabilities in the model's safety layers.",
            style="yellow"
        )

        token_analysis.update(result)

    async def run_compare(self):
        """Run both interactions for comparison"""
        status_bar = self.query_one("#status-bar", StatusBar)
        status_bar.set_status("Running comparison...", "yellow")

        await self.run_normal()
        await asyncio.sleep(0.5)  # Brief pause for UX
        await self.run_advtok()


if __name__ == "__main__":
    app = AdvTokDemoApp()
    app.run()
