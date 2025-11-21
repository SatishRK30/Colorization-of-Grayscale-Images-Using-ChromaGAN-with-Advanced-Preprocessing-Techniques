# Main pipeline linking preprocessing → augmentation → ChromaGAN → evaluation

from preprocessing.apply_all_filters import process_all
from augmentation.horizontal_flip import flip_horizontal
from chromagan.inference import colorize
from evaluation.comparison import evaluate
from chromagan.utils import load_image, save_image

def run_pipeline(input_path, save_path):
    img = load_image(input_path)

    processed = process_all(img)
    flipped = flip_horizontal(processed)

    chroma_out = colorize([processed])
    chroma_out_flip = colorize([flipped])

    save_image(save_path + "_processed.png", chroma_out[0])
    save_image(save_path + "_flipped.png", chroma_out_flip[0])
