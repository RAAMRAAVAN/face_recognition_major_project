console.log("working2");

const image_slider_url = "http://127.0.0.1:8000/api2/v1/imageslider/";

console.log(image_slider_url);
const quick_links_url = "http://127.0.0.1:8000/api2/v1/quicklinks/";

const objectives_of_sikkim_university_url =
  "http://127.0.0.1:8000/api2/v1/Objectives_of_Sikkim_University/";
// Defining async function
async function getimagesliderapi(url) {
  // Storing response
  let response = await fetch(url);
  // Storing data in form of JSON
  const data = await response.json();
  // console.log(data[0].image)
  let sliderimage;
  let image = "";
  let imagebutton = "";
  sliderimage = document.getElementById(`sliderimage`);
  sliderbutton = document.getElementById(`sliderbutton`);
  data.map((slider, index) => {
    imagebutton += `<button type="button" data-bs-target="#carouselExampleCaptions" data-bs-slide-to="${index}" ${
      index == 0 ? "class='active' aria-current='true'" : ""
    }
        aria-current="true" aria-label="Slide ${index + 1}"></button>`;

    image += `<div class="carousel-item ${index == 0 ? "active" : ""}">
        <img src="${
          slider.image
        }" width="200" height="300" class="d-block w-100" alt="..." />
        <div class="carousel-caption slide_elements">
          <p>
          ${slider.about}
          </p>
        </div>
      </div>`;
  });
  // console.log(image);
  sliderbutton.innerHTML = imagebutton;
  sliderimage.innerHTML = image;
}

async function getquicklinksapi(url) {
  // Storing response
  let response = await fetch(url);
  // Storing data in form of JSON
  const data = await response.json();
  // console.log(data);
  let quick_links_collection = "";
  let quick_links = document.getElementById(`quick_links`);
  data.map((quicklink, index) => {
    quick_links_collection += `<li class="link"><a href="${quicklink.file}" class="underline" target="_blank"> <span>${quicklink.text}</span></a></li>`;
  });
  // console.log(quick_links_collection)
  if (quick_links_collection !== undefined)
    quick_links.innerHTML = quick_links_collection;
}

async function get_objectives_of_sikkim_university_api(url) {
  // Storing response
  let response = await fetch(url);
  // Storing data in form of JSON
  const data = await response.json();
  console.log(data);
  let objectives_of_sikkim_university_collection = "";
  let objectives_of_sikkim_university = document.getElementById(
    `objectives_of_sikkim_university`
  );
  data.map((objective, index) => {
    objectives_of_sikkim_university_collection += `<li class="link mt-3">${objective.objective}</li>`;
  });
  console.log(objectives_of_sikkim_university_collection);
  if (objectives_of_sikkim_university_collection !== undefined)
    objectives_of_sikkim_university.innerHTML =
      objectives_of_sikkim_university_collection;
}

getimagesliderapi(image_slider_url);
getquicklinksapi(quick_links_url);
get_objectives_of_sikkim_university_api(objectives_of_sikkim_university_url);
