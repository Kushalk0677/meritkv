# Security Redactions

The release-preparation credential scan found embedded Hugging Face access
tokens in two historical notebook input cells:

- `experiments/archive/v10_snapshot_202608/semantic_fidelity_colab.ipynb`
- `experiments/archive/v10_snapshot_202608/fidelity_equiv_colab.ipynb`

Only each token literal was replaced with `hf_REDACTED`. Notebook code around
the assignment, outputs, metadata, and all scientific records were preserved.
This is the sole intentional byte-level change inside the V10 source snapshot.
The original credentials are not required for reproduction; users should
authenticate through their environment or `hf auth login`.

No additional Hugging Face, OpenAI, GitHub, AWS, bearer-token, or private-key
patterns were found in the expanded release tree after redaction. Historical
archives were also expanded so their text contents were included in the scan.
