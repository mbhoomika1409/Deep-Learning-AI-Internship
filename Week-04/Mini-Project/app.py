"""Waste Image Classifier: simple Gradio demo.  Run: python app.py"""
import gradio as gr

from predict import predict

THRESHOLD = 0.6   # below this score the app says "Not sure" (chosen on the test table)

LIMITATIONS = """
### Limitations: please read
- **A model score is not a guarantee.** It is how strongly the model favours one class,
  not the probability that it is right. On the test set, 6 of 352 photos were wrong
  with a score of 90% or higher.
- Overall test accuracy is about **79.5%**. Glass is often confused with metal, and plastic with glass.
- It was trained on only **2,357 photos** (TrashNet), mostly **one object on a white
  background**. Cluttered real-world photos, dirty or crushed items, and unusual lighting
  will do worse.
- It knows only **5 classes**: cardboard, glass, metal, paper, plastic. For anything
  else (food, people, a different material) it still picks one of these five.
- Do not use it for real waste-sorting or safety decisions.
"""


def classify(img):
    if img is None:
        return None, "Please upload an image first."
    scores = predict(img)
    top = max(scores, key=scores.get)
    conf = scores[top]
    if conf < THRESHOLD:
        msg = (f"### Not sure\nThe highest score is only {conf:.0%} (for **{top}**), "
               f"below the {THRESHOLD:.0%} threshold. Try a clearer photo of a single "
               "item on a plain background.")
    else:
        msg = (f"### Prediction: {top}\nModel score: **{conf:.0%}**. "
               "Remember: a high score does not guarantee the answer is correct.")
    return scores, msg


demo = gr.Interface(
    fn=classify,
    inputs=gr.Image(type="pil", label="Upload a photo of a waste item"),
    outputs=[gr.Label(num_top_classes=3, label="Top 3 predictions (model scores)"),
             gr.Markdown()],
    title="Waste Image Classifier",
    description="Upload a photo. A small CNN trained on TrashNet predicts: "
                "cardboard, glass, metal, paper or plastic.",
    article=LIMITATIONS,
)

if __name__ == "__main__":
    demo.launch()
