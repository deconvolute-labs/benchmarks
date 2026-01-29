# SQuAD Subset (v1.1 Validation)

This directory contains a randomly sampled subset of 300 samples from the validation set of **The Stanford Question Answering Dataset (SQuAD) v1.1**.

**Source:** [https://rajpurkar.github.io/SQuAD-explorer/](https://rajpurkar.github.io/SQuAD-explorer/)

## Download Dataset
You can download the sampled dataset like this:

```bash
uv run python scripts/fetch_squad_data.py
```

This will create the file `squad_subset_300.json` in `data/corpus/` and allows you to generate datasets with malicious payloads using the CLI.
See the Readme for more details on how to use it.


## Design Choices & Methodology

This dataset serves as the "Clean Corpus" baseline for the Deconvolute RAG Benchmark. Below are the specific justifications for these choices.

### 1. Why SQuAD?
We selected SQuAD v1.1 because it guarantees that every sample contains a high-quality Context paragraph (from Wikipedia) that directly contains the answer.

* **Cleaner Pipeline:** SQuAD provides clean, self-contained text ready for chunking.
* **Focus on Integrity, not Retrieval:** Our benchmark focuses on *safety* (ignoring malicious instructions), not *retrieval performance*. SQuAD eliminates "retrieval noise" (where the answer isn't found), ensuring that every failure we see is due to the attack, not poor document quality.

### 2. Why SQuAD v1.1 (vs. v2.0)?
SQuAD v2.0 introduces "unanswerable" questions. We explicitly chose **v1.1 (Answerable only)** to ensure the LLM is always incentivized to read and process the context.

* **Forcing Attention:** If we used v2.0, the LLM might correctly ignore a context chunk because it deems the question unanswerable. In a security test, we *want* the LLM to attend to the context so we can prove that the Canary Token successfully intercepted that attention.
* **Clearer Signal:** In v1.1, the "Benign" baseline is always a correct answer. If the model refuses to answer or deviates, we know it is a direct result of the attack injection.

### 3. Why the Validation Split?
We use the Validation split because it is a standard, high-quality reference set used for evaluation in academia.

* **Data Integrity:** The "Test" splits of many datasets often have withheld labels (for leaderboards). The validation split provides open, verified ground truth.
* **Avoids Training Bias:** While modern LLMs have likely seen all of Wikipedia, using the validation split is the standard practice for ensuring we aren't evaluating on the primary training objective of simpler models.

### 4. Why 300 Samples?
We selected **300 samples** as the optimal balance between statistical significance and engineering velocity.

* **Statistical Confidence:** 300 samples provide a sufficiently low margin of error for a defense benchmark. A change in detection rate of >1-2% is clearly visible, separating signal from noise.
* **Feedback Loop:** This volume allows the full benchmark to run in minutes rather than hours, enabling rapid iteration on defense logic without incurring massive API costs or latency.


## References

Rajpurkar, Pranav, Jian Zhang, Konstantin Lopyrev, and Percy Liang. “SQuAD: 100,000+ Questions for Machine Comprehension of Text.” arXiv:1606.05250. Preprint, arXiv, October 11, 2016. https://doi.org/10.48550/arXiv.1606.05250.
