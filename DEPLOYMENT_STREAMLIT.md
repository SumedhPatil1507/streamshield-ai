# StreamShield AI — Streamlit Cloud Deployment Guide

## Quick Deployment Instructions

### 1. Deploy to Streamlit Cloud

The StreamShield AI app is now ready for deployment to Streamlit Cloud. Follow these steps:

1. **Go to Streamlit Cloud**: https://share.streamlit.io/deploy

2. **Repository Information**:
   - **Repository URL**: `https://github.com/SumedhPatil1507/streamshield-ai`
   - **Branch**: `main`
   - **Main file path**: `app.py`

3. **App Configuration**:
   - **App URL**: `streamshield-ai.streamlit.app` (or your preferred subdomain)
   - Click "Check availability" to verify your domain

4. **Advanced Settings** (optional):
   - No additional configuration needed
   - The app runs in offline mode by default (no external API dependencies)

5. **Click "Deploy"**

The deployment will automatically:
- Install dependencies from `requirements-streamlit.txt`
- Use the Streamlit configuration from `.streamlit/config.toml`
- Launch the app in offline mode with synthetic data

---

## Deployment Files Added

### `.streamlit/config.toml`
- Custom dark theme configuration
- Streamlit server settings for production deployment
- Client-side error handling enabled

### `requirements-streamlit.txt`
- Streamlit Cloud-specific Python dependencies
- Excludes Kafka, PostgreSQL, and WebRTC dependencies not needed for the dashboard
- Includes core inference and visualization libraries

---

## App Features on Streamlit Cloud

### 🛡️ Live Moderation Studio
- Interactive scoring with preset scenarios
- Real-time risk visualization with Plotly gauges
- 128-dimensional vector space projections

### ⚡ Stream Pipeline Simulator
- High-frequency streaming generator
- Real-time quarantine action queue
- Moderator override capabilities

### 📊 Analytics & Audit Logs
- Historical compliance metrics
- 1-click CSV audit trail export
- Performance dashboard

### 🚀 Interactive Benchmarks
- Comparative performance matrix (CPU vs CUDA vs TensorRT)
- Concurrency scaling analysis (1-100 streams)
- Memory footprint breakdown
- Live stress testing with percentile latency analysis

---

## Offline Mode Configuration

The app is configured to run in **offline mode** by default:
- `STREAMSHIELD_OFFLINE=1` is set in `app.py`
- Uses synthetic images instead of external HTTP requests
- No external API keys or credentials required
- Suitable for air-gapped environments

---

## Expected Deployment Time

- **Initial deployment**: 2-3 minutes (dependency installation)
- **Subsequent deployments**: 30-60 seconds
- **Cold start**: 10-15 seconds (first user visit)

---

## Resource Requirements

Streamlit Cloud provides:
- **CPU**: 1 vCPU minimum
- **Memory**: 512 MB minimum (recommended: 1 GB)
- **Storage**: 1 GB included

The StreamShield AI dashboard is optimized for these resources:
- Memory footprint: ~300-400 MB
- CPU usage: Low (synthetic data processing)
- No GPU required (simulation mode)

---

## Customization

### Adding External APIs

To connect to real inference engines, modify the environment variables:

```python
# In app.py, comment out or remove:
os.environ.setdefault("STREAMSHIELD_OFFLINE", "1")

# Add your API endpoints and credentials
os.environ["STREAMSHIELD_INFERENCE_API"] = "https://your-api.example.com"
os.environ["STREAMSHIELD_API_KEY"] = "your-api-key"
```

### Adding Real-Time Data Sources

For live WebRTC/RTSP integration:
1. Deploy the backend API server separately (e.g., on Render, Railway, or Kubernetes)
2. Update the app to connect to your backend endpoint
3. Use `requirements.txt` instead of `requirements-streamlit.txt` for full dependencies

---

## Troubleshooting

### packages.txt Installation Errors

**Error**: `Unable to locate package` errors for comment lines

**Solution**: The `packages.txt` file should only contain package names, one per line, without comments. For the StreamShield dashboard, no system packages are needed since it runs in simulation mode. The file has been removed from the repository.

### Deployment Fails

**Error**: `ModuleNotFoundError: No module named 'src'`

**Solution**: Ensure the repository structure is preserved. The app expects:
```
streamshield-ai/
├── app.py
├── src/
│   ├── models/
│   └── streaming/
└── requirements-streamlit.txt
```

### App Crashes on Startup

**Error**: Import errors or missing dependencies

**Solution**: Check that `requirements-streamlit.txt` includes all necessary packages. The current version includes:
- streamlit
- plotly
- pandas
- onnxruntime
- transformers
- Pillow
- numpy

### Slow Performance

**Solution**: The app uses synthetic data for fast performance. If experiencing slowness:
1. Check Streamlit Cloud status (https://status.streamlit.io)
2. Reduce the number of concurrent simulations
3. Use the lightweight "Live Studio" tab instead of "Stream Simulator"

---

## Next Steps

After successful deployment:

1. **Share your app**: Copy the URL from Streamlit Cloud and share with stakeholders
2. **Monitor usage**: Streamlit Cloud provides basic analytics
3. **Customize branding**: Modify the CSS in `app.py` to match your brand
4. **Add real inference**: Connect to your backend API for production use
5. **Scale up**: For high-traffic deployments, consider the paid Streamlit Cloud tier

---

## Support

For issues or questions:
- GitHub Issues: https://github.com/SumedhPatil1507/streamshield-ai/issues
- Streamlit Cloud Docs: https://docs.streamlit.io/streamlit-cloud
- Email: support@streamshield.ai (hypothetical)

---

**Last Updated**: October 2026
**Deployment Version**: 3.2.0-PROD
