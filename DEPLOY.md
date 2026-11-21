# Put InkVerify on GitHub and host it

GitHub stores the code. A phone can use the app only when something runs it, so connect the
repo to a Python host. Free tiers change - check each site's current limits.

## 1. GitHub
```bash
git init
git add .
git commit -m "InkVerify"
git branch -M main
git remote add origin https://github.com/<you>/inkverify.git
git push -u origin main
```
- `.gitignore` already keeps out `.venv`, `uploads` and `data/signatures` (personal data).
- The trained model `models/inkverify_model.joblib` IS committed so the hosted app can score.
  If it was trained on demo data the app says so on every result. Remove the file to hide it.

## 2a. Render (simplest)
1. render.com -> New -> **Blueprint** -> choose the repo (it reads `render.yaml`),
   or New -> Web Service with build `pip install -r requirements.txt` and start
   `gunicorn app:app --workers 1 --threads 4 --timeout 120 --bind 0.0.0.0:$PORT`.
2. In the service's Environment tab set **INKVERIFY_PASSWORD** so only people you tell can open it.
3. Open the `https://....onrender.com` link on your phone. Free services sleep when idle; the first load can be slow.

## 2b. Hugging Face Spaces (Docker)
New Space -> SDK **Docker** -> upload the files. The `Dockerfile` serves on port 7860.
Add `INKVERIFY_PASSWORD` under Settings -> Variables and secrets.

## Checklist before sharing the link
- Set a password (the app has no accounts).
- Know that anyone with access can upload images to it; do not use real people's signatures without consent.
- Phone camera upload needs https, which these hosts provide.
