async function sendFile() {
  const file = document.getElementById("file").files[0];

  if (!file) {
    console.error("No file selected!");
    return; // Exit if no file is selected
  }

  const formData = new FormData();
  formData.append("file", file); // Append the file to FormData

  try {
    // Replace 'http://localhost:5000/predict' with your API endpoint
    const response = await axios.post(
      "http://localhost:8000/predict",
      formData,
      {
        headers: {
          "Content-Type": "multipart/form-data", // Set content type for file upload
        },
      }
    );

    const output = document.getElementById("output");
    output.innerHTML = id2label[response.data.prediction];
  } catch (error) {
    console.error("Error uploading file:", error); // Handle error response
  }
}
