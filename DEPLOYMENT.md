# 🚀 Website Deployment Guide

## 📋 **Prerequisites**

1. **Google Account** (Gmail)
2. **Credit Card** (for GCP verification - won't be charged on free tier)
3. **Computer** with internet connection

## 🔧 **Step 1: Setup Google Cloud Platform**

### 1.1 Create GCP Account
- Go to [Google Cloud Console](https://console.cloud.google.com/)
- Sign in with your Google account
- Accept terms and conditions
- **Important**: You'll need to add a credit card for verification

### 1.2 Create New Project
```bash
# In GCP Console:
1. Click "Select a project" at the top
2. Click "New Project"
3. Project name: "whatsapp-analyzer-[YOUR-NAME]"
4. Click "Create"
```

### 1.3 Enable Billing
```bash
# In GCP Console:
1. Go to "Billing" in left menu
2. Click "Link a billing account"
3. Create new billing account
4. Add your credit card (required for verification)
```

### 1.4 Enable App Engine
```bash
# In GCP Console:
1. Go to "App Engine" in left menu
2. Click "Create Application"
3. Choose region: "us-central" (recommended)
4. Click "Create app"
```

## 💻 **Step 2: Install Google Cloud CLI**

### 2.1 Download & Install
- **Windows**: Download from [Google Cloud SDK](https://cloud.google.com/sdk/docs/install)
- **macOS**: Use Homebrew: `brew install google-cloud-sdk`
- **Linux**: Follow [official guide](https://cloud.google.com/sdk/docs/install)

### 2.2 Verify Installation
```bash
gcloud --version
```

## 🔐 **Step 3: Login & Setup**

### 3.1 Login to GCP
```bash
gcloud auth login
```
- This will open your browser
- Sign in with your Google account
- Grant permissions

### 3.2 Set Project
```bash
# Replace [YOUR-PROJECT-ID] with your actual project ID
gcloud config set project [YOUR-PROJECT-ID]
```

### 3.3 Verify Project
```bash
gcloud config get-value project
```

## 📁 **Step 4: Prepare Your Project**

### 4.1 Install Dependencies
```bash
# Install Node.js dependencies
npm install

# Install Python dependencies (if you have Python installed)
pip install -r requirements.txt
```

### 4.2 Test Locally
```bash
npm start
```
- Open http://localhost:3000
- Verify everything works

## 🚀 **Step 5: Deploy to GCP**

### 5.1 Deploy Application
```bash
npm run deploy
```

**What happens:**
- GCP will build your application
- Upload all files
- Deploy to App Engine
- Show deployment progress

### 5.2 Wait for Deployment
```bash
# This may take 5-10 minutes
# You'll see progress like:
# Updating service [default]...done.
# Setting traffic split for service [default]...done.
# Deployed service [default] to [https://your-app.appspot.com]
```

### 5.3 Open Your Website
```bash
npm run open
```
- This will open your live website in browser
- Your URL will be: `https://[YOUR-PROJECT-ID].appspot.com`

## 🔍 **Step 6: Monitor & Manage**

### 6.1 View Logs
```bash
npm run logs
```

### 6.2 Check Status
```bash
gcloud app describe
```

### 6.3 Update Website
```bash
# After making changes:
npm run deploy
```

## 🌐 **Your Website URLs**

- **Production**: `https://[YOUR-PROJECT-ID].appspot.com`
- **Admin**: `https://console.cloud.google.com/appengine`

## 💰 **Costs & Billing**

### Free Tier (First 12 months):
- **App Engine**: 28 instance hours/day
- **Storage**: 5GB
- **Data Transfer**: 1GB/day

### After Free Tier:
- **App Engine**: ~$0.05/hour
- **Storage**: ~$0.02/GB/month
- **Data Transfer**: ~$0.12/GB

## 🆘 **Troubleshooting**

### Common Issues:

1. **"Billing not enabled"**
   - Go to GCP Console → Billing
   - Link a billing account

2. **"Project not found"**
   - Verify project ID: `gcloud config get-value project`
   - Create project if needed

3. **"Permission denied"**
   - Run: `gcloud auth login`
   - Grant necessary permissions

4. **"Build failed"**
   - Check logs: `npm run logs`
   - Verify all dependencies are installed

### Get Help:
- **GCP Documentation**: [cloud.google.com/docs](https://cloud.google.com/docs)
- **GCP Support**: [cloud.google.com/support](https://cloud.google.com/support)
- **Community**: [Stack Overflow](https://stackoverflow.com/questions/tagged/google-cloud-platform)

## 🎉 **Congratulations!**

Your WhatsApp Chat Analyzer is now live on the internet! 

**Next Steps:**
1. Share your website URL with others
2. Test all features on the live site
3. Monitor usage in GCP Console
4. Make updates as needed

---

**Need Help?** Check the troubleshooting section or contact GCP support.
