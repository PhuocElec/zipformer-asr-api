# Weights Directory

This folder stores local model weights and related assets used by the application at runtime. Keep any large or private model files here so they are available to the app without requiring a network download.

What to place here
- Model checkpoints (e.g., `.pt`, `.bin`, `.onnx`).
- Model configs and metadata (e.g., `.yaml`, `.json`).
- Tokenizer files or symbol tables (e.g., `bpe.model`, `tokens.txt`).

Recommended layout (example)
```
app/weights/
  your-model-name/
    model.pt
    config.yaml
    tokens.txt
```

Notes
- Large binaries are typically not committed to version control; ensure your `.gitignore` excludes them if needed.
- If building a Docker image, either copy the required weights into the image or mount this directory as a volume at runtime.
- Update your app configuration to point to the correct paths inside `app/weights/` if the code expects specific filenames or subfolders.

This README exists to keep the directory in version control and to document its purpose.
diff --git a/c:\Project\Tel4VN\zipformer-asr-api\app/weights/README.md b/c:\Project\Tel4VN\zipformer-asr-api\app/weights/README.md
--- a/c:\Project\Tel4VN\zipformer-asr-api\app/weights/README.md
+++ b/c:\Project\Tel4VN\zipformer-asr-api\app/weights/README.md
@@ -0,0 +1,24 @@
+# Weights Directory
+
+This folder stores local model weights and related assets used by the application at runtime. Keep any large or private model files here so they are available to the app without requiring a network download.
+
+What to place here
+- Model checkpoints (e.g., `.pt`, `.bin`, `.onnx`).
+- Model configs and metadata (e.g., `.yaml`, `.json`).
+- Tokenizer files or symbol tables (e.g., `bpe.model`, `tokens.txt`).
+
+Recommended layout (example)
+```
+app/weights/
+  your-model-name/
+    model.pt
+    config.yaml
+    tokens.txt
+```
+
+Notes
+- Large binaries are typically not committed to version control; ensure your `.gitignore` excludes them if needed.
+- If building a Docker image, either copy the required weights into the image or mount this directory as a volume at runtime.
+- Update your app configuration to point to the correct paths inside `app/weights/` if the code expects specific filenames or subfolders.
+
+This README exists to keep the directory in version control and to document its purpose.
