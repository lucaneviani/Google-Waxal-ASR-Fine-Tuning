/* ==========================================================================
   DATA SCIENCE PORTFOLIO - CASE STUDY INTERACTIVE LOGIC
   ========================================================================== */

document.addEventListener('DOMContentLoaded', () => {
    initCounters();
    initPipelineExplorer();
});

/* --------------------------------------------------------------------------
   1. KPI NUMBER COUNTERS (ANIMATED ON SCROLL)
   -------------------------------------------------------------------------- */
function initCounters() {
    const counters = document.querySelectorAll('.counter');
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const counter = entry.target;
                const target = parseFloat(counter.getAttribute('data-target'));
                const isFloat = counter.getAttribute('data-target').includes('.');
                const duration = 1500;
                const stepTime = 20;
                const steps = duration / stepTime;
                const increment = target / steps;
                let current = 0;

                const timer = setInterval(() => {
                    current += increment;
                    if (current >= target) {
                        counter.innerText = isFloat ? target.toFixed(1) : Math.floor(target);
                        clearInterval(timer);
                    } else {
                        counter.innerText = isFloat ? current.toFixed(1) : Math.floor(current);
                    }
                }, stepTime);

                observer.unobserve(counter);
            }
        });
    }, { threshold: 0.5 });

    counters.forEach(counter => observer.observe(counter));
}

/* --------------------------------------------------------------------------
   2. ARCHITECTURE PIPELINE EXPLORER
   -------------------------------------------------------------------------- */
const pipelineData = {
    "step1": {
        badge: "STEP 01 &bull; DATA INGESTION",
        title: "Streaming Audio from Hugging Face & 16 kHz Normalization",
        desc: "Instead of downloading heavy audio archives to local disk storage, the pipeline streams dataset files directly from Hugging Face servers. Every recording is automatically resampled to a standardized frequency of 16,000 Hz, which is the required acoustic standard for neural network audio encoders.",
        code: `def load_speech_dataset(language="lug", split="train"):
    # Stream dataset directly from Hugging Face cloud servers
    ds = load_dataset("google/WaxalNLP", name=f"{language}_asr", split=split, streaming=True)
    # Standardize all audio files to 16 kHz for neural network input
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    return ds

train_dataset = load_speech_dataset(language="lug", split="train")`,
        takeaway: "<strong><i class='fa-solid fa-check-circle'></i> Key Benefit:</strong> Efficiently processes large-scale audio datasets streamed directly from Hugging Face in cloud environments."
    },
    "step2": {
        badge: "STEP 02 &bull; PROMPT FORMATTING",
        title: "Aligning Audio with Textual Transcriptions",
        desc: "Modern speech LLMs process audio as a conversation. The pipeline transforms each raw acoustic array into a user question ('Transcribe this speech') and pairs it with the official text transcription as the target answer. This allows the model to learn the connection between speech sounds and written words.",
        code: `def format_example(example):
    conversation = [
        {"role": "user", "content": [
            {"type": "audio", "audio_url": "dummy"},
            {"type": "text", "text": "Trascrivi questo audio."}
        ]},
        {"role": "assistant", "content": example['transcription']}
    ]
    # Format as a standard prompt and prepare audio features
    text = processor.apply_chat_template(conversation, tokenize=False)
    inputs = processor(text=text, audio=example["audio"]["array"], sampling_rate=16000)
    
    result = {k: v[0] for k, v in inputs.items()}
    result["labels"] = result["input_ids"].clone()
    return result`,
        takeaway: "<strong><i class='fa-solid fa-check-circle'></i> Key Benefit:</strong> Structures multi-modal audio arrays and textual transcriptions into a unified format required by neural network language models."
    },
    "step3": {
        badge: "STEP 03 &bull; MODEL ADAPTATION",
        title: "Lightweight Fine-Tuning with LoRA",
        desc: "Retraining all 1.7 billion parameters of Qwen3 from scratch requires expensive supercomputer clusters. Instead, I froze the original model to preserve its general intelligence, and attached small, trainable 'adapter' layers (LoRA). I trained only 1.8% of the weights, reducing GPU memory requirements by 80%.",
        code: `from peft import LoraConfig, get_peft_model

# Configure lightweight LoRA adapters
lora_config = LoraConfig(
    r=32,                   # Capacity of adapter layers
    lora_alpha=64,          # Scaling factor
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none"
)

# Attach adapters to the frozen base model
model.enable_input_require_grads()
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Result: trainable params: 31M || all params: 1.73B || trainable%: 1.81%`,
        takeaway: "<strong><i class='fa-solid fa-check-circle'></i> Key Benefit:</strong> Adapts a 1.7-billion parameter foundational speech model cost-effectively on standard cloud GPU resources."
    }
};

function initPipelineExplorer() {
    const nodes = document.querySelectorAll('.pipe-node');
    nodes.forEach(node => {
        node.addEventListener('click', () => {
            nodes.forEach(n => n.classList.remove('active'));
            node.classList.add('active');
            const stepId = node.getAttribute('data-node');
            const data = pipelineData[stepId];

            const panel = document.getElementById('nodeDetails');
            panel.style.opacity = '0.3';
            setTimeout(() => {
                document.getElementById('detailStepBadge').innerText = data.badge;
                document.getElementById('detailTitle').innerText = data.title;
                document.getElementById('detailDesc').innerText = data.desc;
                document.getElementById('detailCode').innerText = data.code;
                document.getElementById('detailTakeaway').innerHTML = data.takeaway;
                panel.style.opacity = '1';
            }, 200);
        });
    });
}

/* --------------------------------------------------------------------------
   3. COPY CODE FUNCTIONALITY
   -------------------------------------------------------------------------- */
window.copyCode = function() {
    const codeEl = document.getElementById('detailCode');
    navigator.clipboard.writeText(codeEl.innerText).then(() => {
        const btn = document.querySelector('.btn-copy');
        const originalText = btn.innerHTML;
        btn.innerHTML = '<i class="fa-solid fa-check" style="color: var(--accent-green);"></i> Copied!';
        setTimeout(() => {
            btn.innerHTML = originalText;
        }, 2000);
    });
};
