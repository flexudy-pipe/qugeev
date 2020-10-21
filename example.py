from evaluation import Evaluator
from time import time

# Load the evaluator. Super simple!
# Just go to huggingface.co and select a model. I would recommend question-answering models
# like the ones trained on squad.
# If you have no idea, just select a model that corresponds to your target language

# For simplicity reasons, I hardcoded a lot of parameters like n-gram size, the evaluation metric (which is F1 rouge),
# the context window and number of GPUs. Feel free to modify the script if you want to. But I think the default setting
# is good enough for questions-answer generation evaluation.

evaluator = Evaluator(hugging_face_model_name="distilbert-base-uncased-distilled-squad",
                      tokenizer_name="distilbert-base-uncased",
                      cuda_is_available=False)

start = time()

questions = ["What is the term for a family of team sports?", "What is football?"]

answers = ["Football",
           "Football is a family of team sports that involve, to varying degrees, kicking a ball to score a goal."]

# I intentionally generated the two questions from the same text.

contexts = ["Football is a family of team sports that involve, to varying degrees, kicking a ball to score a goal.",
            "Football is a family of team sports that involve, to varying degrees, kicking a ball to score a goal."]

score = evaluator.evaluate_question_answer_pairs(questions, answers, contexts, verbose=True)

end = time()

print("Took {} seconds to evaluate".format(end - start))
