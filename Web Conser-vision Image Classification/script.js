// Referencias a elementos HTML
const fileInput = document.getElementById("fileInput");
const preview = document.getElementById("preview");
const resultado = document.getElementById("resultado");
const resultados = document.getElementById("resultados");

// Mapeo de especies
const MAP_SPECIES = {
  0: "antelope_duiker",
  1: "bird",
  2: "blank",
  3: "civet_genet",
  4: "hog",
  5: "leopard",
  6: "monkey_prosimian",
  7: "rodent",
};

let model = null;

// Función para cargar el modelo de TensorFlow utilizando async/await
(async () => {
  try {
    console.log("Cargando modelo...");
    model = await tf.loadLayersModel("model.json"); // Carga modelo de TensorFlow
    console.log("¡Modelo cargado!");
  } catch (error) {
    console.error("Error al cargar el modelo:", error);
    alert("No se pudo cargar el modelo. Intenta de nuevo más tarde.");
  }
})();

// Evento en la selección de archivos
fileInput.addEventListener("change", function () {
  resultado.innerText = "";
  resultados.innerHTML = "";
  const reader = new FileReader();

  reader.addEventListener("load", function () {
    preview.src = reader.result;
    predictImage(preview); // Realiza predicción de la imagen cargada
  });

  reader.readAsDataURL(fileInput.files[0]);
});

// Función para realizar la predicción de la imagen
const predictImage = async (image) => {
  await image.decode(); // Decodifica la imagen
  const tensorImg = tf.browser
    .fromPixels(image)
    .resizeNearestNeighbor([75, 75]) // Redimensionar la imagen
    .toFloat()
    .div(tf.scalar(255))
    .expandDims();
  const pred = await model.predict(tensorImg).data(); // Realiza predicción con el modelo

  let i = pred.indexOf(Math.max(...pred)); // Obtiene el índice de la predicción más alta
  resultado.innerText = `Predicción: ${MAP_SPECIES[i]}(${pred[i]})`;

  pred.forEach((e, i) => {
    resultados.innerHTML += `<tr>
      <td>${MAP_SPECIES[i]}</td>
      <td><progress min='0' value='${e}' max='1'></progress></td>
      <td>${Math.round(e * 100) / 100}</td>
    </tr>`;
  });
};
