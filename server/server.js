require("dotenv").config();
const express = require("express");
const cors = require("cors");
const axios = require("axios");
const mongoose = require("mongoose");

const app = express();
const PORT = 8000;

// Middleware
app.use(cors());
app.use(express.json());

mongoose.connect(process.env.MONGO_URI)
  .then(() => console.log("MongoDB connected (Railway)"))
  .catch(err => console.error("MongoDB error:", err.message));


// 🔹 Schema to store predictions
const PredictionSchema = new mongoose.Schema({
  features: Array,
  prediction: Number,
  result: String,
  createdAt: {
    type: Date,
    default: Date.now
  }
});

const Prediction = mongoose.model("Prediction", PredictionSchema);

// Health check
app.get("/", (req, res) => {
  res.json({ message: "Node backend running with MongoDB" });
});

// 🔹 Predict + store in DB
app.post("/predict", async (req, res) => {
  try {
    // Call ML API
    const response = await axios.post(
      "http://localhost:8002/predict",
      req.body
    );

    const { prediction, result } = response.data;

    // Save to MongoDB
    const record = new Prediction({
      features: req.body.features,
      prediction,
      result
    });

    await record.save();

    res.json(response.data);
  } catch (error) {
    console.error(error.message);
    res.status(500).json({ error: "Prediction failed" });
  }
});

// 🔹 Optional: view history
app.get("/history", async (req, res) => {
  const data = await Prediction.find().sort({ createdAt: -1 });
  res.json(data);
});

// Server start (ALWAYS LAST)
app.listen(PORT, () => {
  console.log(`Node backend running at http://localhost:${PORT}`);
});
