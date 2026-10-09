# Enterprise RAG frontend

React/Vite frontend for the existing FastAPI API. It keeps the backend routes unchanged:

- `POST /documents` for PDF, DOCX, and TXT indexing
- `POST /chat` for conversation-aware questions

## Run locally

```powershell
cd frontend
npm install
npm run dev
```

Optional environment variables:

```dotenv
VITE_API_URL=http://127.0.0.1:8000
VITE_API_KEY=
```

If the API uses an `API_KEY`, it can also be entered in the UI for the current browser session. Do not commit a production API key into `VITE_API_KEY`, because Vite variables are bundled into browser JavaScript.
