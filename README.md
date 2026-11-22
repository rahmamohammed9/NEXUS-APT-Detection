# NEXUS — APT Detection & Investigation

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Create a GitHub repository.
2. Upload the contents of this folder to the repository root.
3. Open https://share.streamlit.io/ and connect the repository.
4. Set the main file path to `app.py`.
5. Deploy.

## Input
Upload a CSV exported from the Windows-APT/Wazuh event schema used during training. The app reconstructs the same 21 model inputs from raw event columns.

## Important
This is a research/demo prototype. Its binary target is a weak-label proxy, not independently verified ground truth. Scores are not calibrated probabilities of compromise. Do not use for operational blocking or incident declarations without independent validation.
