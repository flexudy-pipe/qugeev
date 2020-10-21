from data_representation import Document
from haystack.reader.transformers import TransformersReader
from sumeval.metrics.rouge import RougeCalculator
from typing import List
from time import time


class Evaluator:

    def __init__(self,
                 hugging_face_model_name: str = "distilbert-base-uncased-distilled-squad",
                 tokenizer_name: str = "distilbert-base-uncased",
                 cuda_is_available: bool = True):
        cuda_is_available = 0 if cuda_is_available else -1

        self.__reader = TransformersReader(model=hugging_face_model_name,
                                           tokenizer=tokenizer_name,
                                           context_window_size=512,
                                           use_gpu=cuda_is_available)

        self.__rouge = RougeCalculator(stopwords=False)

    def __evaluate_question_answer_pair(self, question: str, answer: str, context: str, identifier: int,
                                        verbose: bool = False) -> float:

        start = None

        if verbose:
            start = time()

        document = Document(identifier, context)

        predictions = self.__reader.predict(question=question, documents=[document], top_k=1)

        predicted_answer = predictions["answers"][0]["answer"]

        score = self.__compute_f1_measure(answer, predicted_answer)

        if verbose:
            end = time()

            print("Question: {}\nPredicted: {}\nGenerated: {}\nScore: {}\nTook {} seconds.\n_____________\n".format(
                question, predicted_answer,
                answer, score,
                end - start))
        return score

    def __compute_f1_measure(self, generated_answer: str, predicted_answer: str) -> float:

        if predicted_answer is None:
            predicted_answer = ""

        rouge_score = self.__rouge.rouge_n(summary=generated_answer, references=predicted_answer, n=1)

        return rouge_score

    def evaluate_question_answer_pairs(self, questions: List[str], answers: List[str], contexts: List[str],
                                       verbose: bool = False) -> float:
        """
        :param questions: A list of N questions
        :param answers: A list of N answers. Answer at index 1 is the answer to the question at index 1 in questions
        :param contexts: A list o N passages used to generate the question. Context at index i belongs to question at i
        :param verbose: Print intermediate results.
        :return: The evaluation metric between 0 and 1.
        """

        if len(questions) != len(answers) != len(contexts):
            raise Exception("Questions, Answers and Context lists must be of equal lengths.")

        question_answer_context_triplet = list(zip(questions, answers, contexts))

        score = 0

        counter = 1

        length = len(questions)

        for question, answer, context in question_answer_context_triplet:
            score += self.__evaluate_question_answer_pair(question, answer, context, counter, verbose)

            if verbose and counter % 1 == 0:
                print("\n> {} % done\n".format((counter / length) * 100))

            counter += 1

        score = score / counter

        print("\n\n[FINAL SCORE] ========> {}\n\n".format(score))

        return score
