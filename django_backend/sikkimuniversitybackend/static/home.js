console.log("working2")
// api url
const api_url = 
      "http://127.0.0.1:8000/api2/v1/imageslider/";
  
// Defining async function
async function getapi(url) {
    
    // Storing response
    let response = await fetch(url);
    
    // Storing data in form of JSON
    const data = response.json();
    // console.log(data[0].image)
    // let sliderimage
    // let image=''
    // sliderimage = document.getElementById(`sliderimage`);
    // for(let i=0 ; i<data.length; i++){
    //     console.log(i)
    //     // image+=`<img src="${data[i].image}" class="d-block w-100" alt="..." />`
    //     image+=`<div class="carousel-item ${i==0?"active":""}">
    //     <img src="${data[i].image}" class="d-block w-100" alt="..." />
    //     <div class="carousel-caption slide_elements">
    //       <p>
    //         Administrative block of sikkim unversity , 6th mile
    //         Tadong , Gangtok
    //       </p>
    //     </div>
    //   </div>`
        
    // }
    // console.log(image)
    // sliderimage.innerHTML = image
}
getapi(api_url);

