"""Waste Image Classifier: Gradio demo (final).  Run: python app.py"""
import glob
import os

import gradio as gr

from predict import predict, predict_all

DEFAULT_THRESHOLD = 0.6
EMOJI = {"cardboard": "📦", "glass": "🍾", "metal": "🥫", "paper": "📄", "plastic": "🧴"}

CSS = """
.gradio-container { max-width: 1150px !important; margin: 0 auto !important; padding: 20px 24px !important; }
#main-row { gap: 36px !important; margin-top: 16px; }
#left-col, #right-col { gap: 18px !important; }

.hero { background: linear-gradient(135deg, #0f9d58 0%, #0b7a6b 100%); border-radius: 20px;
        padding: 36px 32px; text-align: center; margin-bottom: 8px; }
.hero h1 { font-size: 2.4rem; font-weight: 800; margin: 0 0 10px 0; color: #ffffff !important; }
.hero p { font-size: 1.1rem; margin: 0 0 18px 0; color: #e8fff3 !important; line-height: 1.6; }
.chip { display: inline-block; background: rgba(255,255,255,0.20); color: #ffffff !important;
        border-radius: 999px; padding: 6px 16px; margin: 4px 5px; font-size: 0.95rem; font-weight: 600; }

.banner { background: linear-gradient(135deg, #0f9d58 0%, #0b7a6b 100%); border-radius: 16px;
          padding: 22px 28px; margin: 14px 0 20px 0; }
.banner h2 { margin: 0 0 6px 0; font-size: 1.6rem; font-weight: 800; color: #ffffff !important; }
.banner p { margin: 0; font-size: 1rem; color: #e8fff3 !important; line-height: 1.6; }

.sec { font-size: 1.15rem; font-weight: 700; margin: 6px 0 4px 0; }
.note { background: #fff8e6; border: 1px solid #f5d98b; border-radius: 12px;
        padding: 14px 18px; line-height: 1.7; font-size: 0.97rem; margin-bottom: 20px; }
.note, .note * { color: #5c4400 !important; }
.caption { opacity: 0.85; font-size: 0.98rem; line-height: 1.6; margin: 4px 0 8px 2px; }
.footer { text-align: center; opacity: 0.75; font-size: 0.9rem; margin: 26px 0 6px 0; }

.result-card { border-radius: 16px; padding: 24px 28px; line-height: 1.7; }
.result-card, .result-card * { color: #1f2933 !important; }
.result-card .big { font-size: 2rem; font-weight: 800; margin-bottom: 8px; }
.result-card .txt { font-size: 1.05rem; }

.stats { display: flex; flex-wrap: wrap; gap: 18px; margin: 16px 0 10px 0; }
.stat { flex: 1 1 190px; background: #ffffff; border: 1px solid #e3e8ee; border-radius: 16px;
        padding: 20px 16px; text-align: center; box-shadow: 0 2px 10px rgba(0,0,0,0.10); }
.stat .num { font-size: 2rem; font-weight: 800; color: #0f9d58 !important; }
.stat .lbl { font-size: 0.93rem; margin-top: 6px; color: #52606d !important; }

.panel { background: #ffffff; border: 1px solid #e3e8ee; border-radius: 16px; padding: 24px 28px;
         margin: 8px 0 24px 0; line-height: 1.75; box-shadow: 0 2px 10px rgba(0,0,0,0.10); }
.panel, .panel p, .panel li, .panel td, .panel th, .panel div, .panel b { color: #1f2933 !important; }
.panel h3 { margin: 0 0 14px 0; font-size: 1.25rem; color: #0b7a6b !important; }
.panel ul { margin: 0; padding-left: 22px; }
.panel li { margin-bottom: 12px; }

.steps { display: flex; flex-wrap: wrap; gap: 16px; }
.step { flex: 1 1 220px; background: #f1faf5; border: 1px solid #cdeedb; border-radius: 14px; padding: 18px 20px; }
.step b { display: block; font-size: 1.05rem; margin-bottom: 6px; color: #0b7a6b !important; }

.tbl { width: 100%; border-collapse: collapse; font-size: 1rem; }
.tbl th, .tbl td { padding: 12px 14px; text-align: left; border-bottom: 1px solid #e3e8ee; background: #ffffff; }
.tbl th { background: #f1f5f9; }
.tbl td:nth-child(n+2) { white-space: nowrap; }
.tbl tr.best td { background: #e6f4ea; font-weight: 700; }
"""


def banner(title, sub):
    return f'<div class="banner"><h2>{title}</h2><p>{sub}</p></div>'


HERO = (
    '<div class="hero"><h1>♻️ Waste Image Classifier</h1>'
    '<p>Upload a photo of a waste item and a small neural network (CNN) will guess its category.</p>'
    '<span class="chip">📦 Cardboard</span><span class="chip">🍾 Glass</span>'
    '<span class="chip">🥫 Metal</span><span class="chip">📄 Paper</span>'
    '<span class="chip">🧴 Plastic</span></div>'
)

STATS = [("79.5%", "Test accuracy"), ("2,357", "Distinct photos"),
         ("5", "Waste classes"), ("6 of 352", "Wrong with a score of 90% or more")]
STATS_HTML = '<div class="stats">' + "".join(
    f'<div class="stat"><div class="num">{n}</div><div class="lbl">{l}</div></div>' for n, l in STATS
) + "</div>"

STEPS_HTML = (
    '<div class="panel"><h3>How it works</h3><div class="steps">'
    '<div class="step"><b>1. Preprocess</b>Your photo is resized to 128 x 128 pixels and normalised '
    'with the ImageNet mean and standard deviation, exactly as in training.</div>'
    '<div class="step"><b>2. Predict</b>A CNN with 4 convolution blocks, BatchNorm and Dropout '
    'gives a score for each of the 5 classes.</div>'
    '<div class="step"><b>3. Decide</b>If the top score is below the threshold, the app says '
    '"Not sure" instead of guessing.</div></div></div>'
)

TABLE_HTML = (
    '<div class="panel"><h3>Test results (352 photos the model never saw)</h3>'
    '<table class="tbl"><tr><th>Model</th><th>Test accuracy</th></tr>'
    '<tr><td>A: baseline CNN (2 plain conv blocks)</td><td>65.1%</td></tr>'
    '<tr class="best"><td>B: improved CNN (BatchNorm, global pooling, Dropout): used in this app</td><td>79.5%</td></tr>'
    '<tr><td>C: improved CNN + data augmentation</td><td>72.7%</td></tr></table></div>'
)

LIMITS_HTML = (
    '<div class="panel"><h3>⚠️ Limitations: please read</h3><ul>'
    '<li><b>A model score is not a guarantee.</b> It shows how strongly the model favours one class, '
    'not the probability that it is right. On the test set, 6 of 352 photos were wrong with a score of 90% or higher.</li>'
    '<li>Overall accuracy is about <b>79.5%</b>. Glass is often confused with metal, and plastic with glass.</li>'
    '<li>Trained on only <b>2,357 photos</b> (TrashNet), mostly <b>one object on a white background</b>. '
    'Cluttered real-life photos, dirty or crushed items and unusual lighting will do worse.</li>'
    '<li>It knows only <b>5 classes</b>. For anything else (food, people, other materials) it still picks one of these five.</li>'
    '<li>Do not use it for real waste-sorting or safety decisions.</li></ul></div>'
)

EVIDENCE = [
    ("Confusion matrix (test set)", "images/confusion_matrix.png",
     "Rows are the true class and columns are the predicted class. The biggest mistake is glass predicted as metal (17 images), then plastic predicted as glass (12)."),
    ("Training and validation curves", "images/training_curves.png",
     "Top row: loss. Bottom row: accuracy. One column per model. The improved CNN (B) learns best. The validation curves are noisy because the validation set is small (354 images)."),
    ("Do model scores guarantee correctness?", "images/confidence_vs_accuracy.png",
     "Accuracy rises with the score, but not to 100%. Six test photos were wrong with a score of 90% or more, so a high score is not proof."),
    ("Most confident mistakes", "images/failure_examples.png",
     "The test photos the model got wrong with the highest scores. Shiny glass, metal and plastic items look alike on a white background."),
]


def card(title, body, bg, border):
    return (f'<div class="result-card" style="background:{bg};border:2px solid {border};">'
            f'<div class="big">{title}</div><div class="txt">{body}</div></div>')


PLACEHOLDER = card("Upload a photo", "The prediction will appear here.", "#f1f3f5", "#ced4da")
COMPARE_PLACEHOLDER = ('<div class="panel"><h3>Baseline vs improved (same photo)</h3>'
                       'Upload a photo to see what all three models predict for it.</div>')


def compare_html(all_scores):
    rows = ""
    for title, sc in all_scores.items():
        top = max(sc, key=sc.get)
        cls = ' class="best"' if "final" in title else ""
        rows += f"<tr{cls}><td>{title}</td><td>{EMOJI[top]} {top}</td><td>{sc[top]:.0%}</td></tr>"
    return ('<div class="panel"><h3>Baseline vs improved (same photo)</h3>'
            '<table class="tbl"><tr><th>Model</th><th>Prediction</th><th>Score</th></tr>'
            + rows + '</table></div>')


def classify(img, threshold):
    if img is None:
        return PLACEHOLDER, None, COMPARE_PLACEHOLDER
    scores = predict(img)
    top = max(scores, key=scores.get)
    conf = scores[top]
    if conf < threshold:
        html = card("🤔 Not sure",
                    f"The top guess is <b>{top}</b>, but only at <b>{conf:.0%}</b>, below your "
                    f"{threshold:.0%} threshold. Try a clearer photo of a single item on a plain background.",
                    "#fff4e0", "#f5a623")
    else:
        html = card(f"{EMOJI[top]} {top.capitalize()}",
                    f"Model score: <b>{conf:.0%}</b><br>A high score does <b>not</b> guarantee the answer is correct.",
                    "#e6f4ea", "#34a853")
    return html, scores, compare_html(predict_all(img))


def clear():
    return None, PLACEHOLDER, None, COMPARE_PLACEHOLDER


with gr.Blocks() as demo:
    gr.HTML(HERO + STATS_HTML)

    with gr.Tabs():
        with gr.Tab("🔍 Classify"):
            gr.HTML(banner("Classify a photo",
                           "Upload a photo or pick a sample. The prediction appears on the right."))
            with gr.Row(elem_id="main-row"):
                with gr.Column(elem_id="left-col"):
                    gr.HTML('<div class="sec">1. Upload a photo</div>')
                    img = gr.Image(type="pil", label="Photo of a waste item", height=340)
                    thr = gr.Slider(0.3, 0.95, value=DEFAULT_THRESHOLD, step=0.05,
                                    label="'Not sure' threshold",
                                    info="Below this score the app will not give an answer.")
                    with gr.Row():
                        run = gr.Button("Classify", variant="primary")
                        reset = gr.Button("Clear")
                    samples = sorted(glob.glob(os.path.join("examples", "*.jpg")))
                    if samples:
                        gr.Examples(examples=[[s] for s in samples], inputs=[img],
                                    label="Or try a sample photo (from the test set)",
                                    cache_examples=False)
                with gr.Column(elem_id="right-col"):
                    gr.HTML('<div class="sec">2. Result</div>')
                    result = gr.HTML(PLACEHOLDER)
                    label = gr.Label(num_top_classes=5, label="Model scores for all classes")

            compare = gr.HTML(COMPARE_PLACEHOLDER)
            gr.HTML('<div class="note">⚠️ <b>Model scores do not guarantee correctness.</b> '
                    'The model only knows 5 clean classes, so a photo of something else '
                    'can still get a high score. See the <i>About the model</i> tab.</div>')

        with gr.Tab("📈 Evidence"):
            gr.HTML(banner("Evidence: how well does it really work?",
                           "Real results from the test set: 352 photos the model never saw during training."))
            for title, path, caption in EVIDENCE:
                if os.path.exists(path):
                    gr.HTML(f'<div class="sec">{title}</div><div class="caption">{caption}</div>')
                    gr.Image(value=path, interactive=False, show_label=False)

        with gr.Tab("📊 About the model"):
            gr.HTML(banner("About the model", "What it is, how it works, and where it fails."))
            gr.HTML(STEPS_HTML + TABLE_HTML + LIMITS_HTML)

    gr.HTML('<div class="footer">Skill Nexis · Deep Learning Internship · Week 4 Major Project · '
            'Dataset: TrashNet (MIT licence)</div>')

    outs = [result, label, compare]
    run.click(classify, [img, thr], outs)
    img.change(classify, [img, thr], outs)
    thr.release(classify, [img, thr], outs)
    reset.click(clear, None, [img] + outs)

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(primary_hue="emerald", secondary_hue="teal"), css=CSS)
