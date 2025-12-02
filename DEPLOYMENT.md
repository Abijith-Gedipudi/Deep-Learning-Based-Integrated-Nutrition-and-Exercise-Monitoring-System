# Deployment Guide for Render

This guide will help you deploy your Food Tracker Flask application to Render for free.

## Prerequisites

1. A GitHub account
2. Your code pushed to a GitHub repository
3. A Render account (sign up at https://render.com - it's free)

## Step-by-Step Deployment Instructions

### Step 1: Prepare Your Repository

1. **Make sure your code is on GitHub:**
   ```bash
   git add .
   git commit -m "Prepare for Render deployment"
   git push origin main
   ```

2. **Important files that should be in your repo:**
   - `app.py` (main application)
   - `models.py` (ML model code)
   - `requirements.txt` (dependencies)
   - `render.yaml` (Render configuration - optional but helpful)
   - `food101_model_for_inference (1).pth` (your trained model - make sure it's committed)
   - `templates/` folder (all HTML templates)
   - `static/` folder (CSS and uploads directory)

### Step 2: Sign Up for Render

1. Go to https://render.com
2. Click "Get Started for Free"
3. Sign up using your GitHub account (recommended for easy integration)

### Step 3: Create a New Web Service

1. Once logged in, click **"New +"** button in the top right
2. Select **"Web Service"**
3. Connect your GitHub account if you haven't already
4. Select your repository from the list
5. Click **"Connect"**

### Step 4: Configure Your Service

Fill in the following settings:

**Basic Settings:**
- **Name**: `food-tracker-app` (or any name you prefer)
- **Region**: Choose the closest region to your users
- **Branch**: `main` (or your default branch)
- **Root Directory**: Leave empty (or `./` if your app is in root)
- **Runtime**: `Python 3`
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `python app.py`

**Advanced Settings (click "Advanced" to expand):**
- **Environment**: `Python 3`
- **Python Version**: Leave default or select `3.11`

### Step 5: Set Environment Variables

Click on **"Environment"** tab and add:

1. **SECRET_KEY**:
   - Click "Add Environment Variable"
   - Key: `SECRET_KEY`
   - Value: Generate a random secret key (you can use: `python -c "import secrets; print(secrets.token_hex(32))"`)

2. **PORT** (usually auto-set by Render, but you can verify):
   - Key: `PORT`
   - Value: `10000` (Render's default, but your app reads this automatically)

### Step 6: Deploy

1. Scroll down and click **"Create Web Service"**
2. Render will start building your application
3. This process may take 5-10 minutes (especially because PyTorch is large)
4. You can watch the build logs in real-time

### Step 7: Access Your App

1. Once deployment is complete, Render will provide you with a URL like:
   `https://food-tracker-app.onrender.com`
2. Click the URL or copy it to access your application
3. Your app should now be live!

## Important Notes

### Database (SQLite)
- Your app uses SQLite, which works on Render's filesystem
- **Important**: Data may be lost if Render restarts your service (free tier services spin down after inactivity)
- For production, consider upgrading to Render's PostgreSQL (paid) or using a different database service

### Model File Size
- Your PyTorch model file (`food101_model_for_inference (1).pth`) might be large
- Make sure it's committed to your GitHub repo
- If it's too large (>100MB), GitHub may require Git LFS (Large File Storage)

### Free Tier Limitations
- **Spinning down**: Free services spin down after 15 minutes of inactivity
- **First request**: May take 30-60 seconds to wake up
- **Build time**: Limited build time per month
- **Storage**: Ephemeral (data may be lost on restart)

### Troubleshooting

**Build fails:**
- Check the build logs in Render dashboard
- Ensure all dependencies in `requirements.txt` are correct
- Verify Python version compatibility

**App crashes on startup:**
- Check the logs tab in Render dashboard
- Verify the model file path is correct
- Ensure all required files are in the repository

**Model not loading:**
- Verify the model file is committed to GitHub
- Check the file path in `app.py` (line 77)
- The model file name has spaces and parentheses - make sure it matches exactly

**Database issues:**
- SQLite should work, but data is ephemeral
- Check that the `instance/` directory is writable

## Updating Your App

1. Make changes to your code locally
2. Commit and push to GitHub:
   ```bash
   git add .
   git commit -m "Update description"
   git push origin main
   ```
3. Render will automatically detect the push and redeploy
4. You can also manually trigger a deploy from the Render dashboard

## Next Steps

- Monitor your app's performance in the Render dashboard
- Set up custom domain (available on paid plans)
- Consider upgrading to a paid plan for persistent storage and no spin-down
- Add error monitoring (e.g., Sentry)

## Support

If you encounter issues:
1. Check Render's documentation: https://render.com/docs
2. Review your build and runtime logs in the Render dashboard
3. Ensure all environment variables are set correctly

Good luck with your deployment! 🚀

