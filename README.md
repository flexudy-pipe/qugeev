# qugeev
A simple way to evaluate question generation models.

Questions Asking is an important NLP task. Making sure that your model predicts the right answers are predicted is crucial for quality assurance.
I will assume that you have access to a software, library or tool of some kind, that can generate question-answer pairs from text (called the context).

This library uses [deepset's haystack](https://github.com/deepset-ai/haystack) to evaluate the answers your model (tool or software) predicts.

## How to use the library
1. You have a list of sentences (contexts)
2. Use your tool or model to generate questions-answer pairs
3. Feed them into the evaluator
4. We use deepset's haystack to predict a candidate answer for each question and context pair
5. We compair the results (predicted by haystack and the one your model generates) using [semeval](https://github.com/chakki-works/sumeval).
6. The mean F1 ROUGE scores (for each answer pairs) is returned.

## Example
```python
from evaluation import Evaluator

evaluator = Evaluator(hugging_face_model_name="distilbert-base-uncased-distilled-squad",
                      tokenizer_name="distilbert-base-uncased",
                      cuda_is_available=False)

questions = ["What is the term for a family of team sports?", "What is football?"]

answers = ["Football",
           "Football is a family of team sports that involve, to varying degrees, kicking a ball to score a goal."]

contexts = ["Football is a family of team sports that involve, to varying degrees, kicking a ball to score a goal.",
            "Football is a family of team sports that involve, to varying degrees, kicking a ball to score a goal."]

score = evaluator.evaluate_question_answer_pairs(questions, answers, contexts, verbose=True)

```
Check out the `example.py` script.

## Limitations
- Although haystack is an amazing library, there is still a very small chance the results predicted by the library could be incorrect. In the evaluator I only evaluate on the best answer returned by haystack. This means, haystack might even predict the correct answer but this answer could have a lower rank. This might affect the F1 score. For example, in the above example, the second answer is actually "correct", but haystack's top answer is `None`. This leads to a score of `0.33` instead of say `0.99`.

- The results also greatly depend on the model used, the n-grams (i use uni-grams for evaluations) and probably other parameters like the the context window. 

Hence feel free to experiment. I however think that the default parameters are good enough to have a good feel about the performance of your model.

