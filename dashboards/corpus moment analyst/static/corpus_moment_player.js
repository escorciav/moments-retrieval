/* Moment player interface
Logic behind all the buttons and ground-truth slider.
The player has a slider associated with it which is created on-the-fly.

TODO: clean up and wrap "global" variables inside a class/object
*/

let loop = true;

function play_video(id) {
  let player = videojs("player_"+id+"_html5_api")
  //set play time as the current time which is displayed by the 2 element of the slider
  const slider = slider_values('SliderPred_'+id)
  player.currentTime(slider[1])
  if (slider[2]-slider[1] <= 0.02) {
      player.currentTime(slider[0])
  }
  player.play();
}

function pause_video(id) {
  videojs("player_"+id+"_html5_api").pause()
}

function trigger_loop() {
  // change the state of the boolean global variable loop
  loop = !loop;
}

function speed_change(id,direction){
  let player = videojs("player_"+id+"_html5_api");
  let playback_rate = player.playbackRate();
  if (direction == 1){          // speed-up
      if (playback_rate < 10) {
          playback_rate += 0.5;
      };
  }
  else{                         //slow-down
      if (playback_rate > 0.5) {
          playback_rate -= 0.5;
      }
  }
  player.playbackRate(playback_rate);   
};

function reset(id) {
    let player = videojs("player_"+id+"_html5_api")
    loop = true;
    player.playbackRate(1.0);
    // Get string from html with initial interval values and remove special characters
    string = document.getElementById("prediction-show-"+id).innerHTML.replace(/<\/?[^>]+(>|$)/g, "");
    // slit string to obtain float values
    time = string.split(": ")[2].split("-").slice(0,2);
    // construct slider vector
    slider = [parseFloat(time[0]),parseFloat(time[0]),parseFloat(time[1])];
    set_slider_values('SliderPred_1', slider);
    player.currentTime(slider[0]);
    player.play();
};

function open_modal(id){
  let player = videojs("player_"+id+"_html5_api");    
    // set playback rate to 1    
    player.playbackRate(1);                         
    // set starting instant to t_min according to the sliders
    player.currentTime(Math.min(...slider_values("SliderPred_"+id)));   
    player.play();
}

function slider_values(element_id) {
  return document.getElementById(element_id).noUiSlider.get();
};

function set_slider_values(element_id, time) {
    document.getElementById(element_id).noUiSlider.set(time);
};


function create_html_slider(parent_id, element_id, pair, max_value=1,
                             step=0.1, onchange=undefined) {
  let div = document.createElement("div");            // create new div
  div.id  = element_id;                               // associate with id

  document.getElementById(parent_id).appendChild(div);    // append to parent div
  let range = document.getElementById(element_id);    // Retrieve info and use to
                                                        // reference object creation
  let pair_ = [pair[0], pair[0], pair[1]];
  noUiSlider.create(range, {
        animationDuration: 300,
        start: pair_,
        step: step,
        margin: 2 * step,
        connect: true,
        orientation: 'horizontal',
        tooltips: true,
        range: {
            'min': 0,
            'max': max_value
        },
  });

  let connect_progress = range.querySelectorAll('.noUi-connect')[0];
  connect_progress.classList.add('noUi-color-progress');
  document.getElementById(parent_id).appendChild(document.createElement("br"));
};


// App functions
function grab_info() {
    /* this should be a POST, but @escorcia was lazy and decided to
       spit out a new page for each moment */
    let num_predictions = parseInt(document.getElementById("num_predictions").innerHTML);
    let durations = JSON.parse(document.getElementById("durations").innerHTML.replace(/'/g,'\"'));
    let ancillary = JSON.parse(document.getElementById("video_names").innerHTML.replace(/'/g,'\"'));
    video_names   = [];
    for (let key in ancillary){
        if (ancillary.hasOwnProperty(key)) {
            video_names.push(ancillary[key]);
        };
    };
    let predictions = []
    for (let i = 1; i <= num_predictions; i++) {
        predictions.push([])
        for (let j = 0; j < 2; j++) {
          let element_id = `prediction${i}-${j}`
          let time = parseFloat(document.getElementById(element_id).innerHTML);
          predictions[predictions.length - 1].push(time)
        };
    };
    $("#DataExchange").remove();
    return [predictions, durations,video_names];
}

/* DOM and App
Glue all the hacks
TODO: this will be much cleaner after modularizing all the components
Short story of this web-app:
The server spits out the whole page each time the user results for a given
moment. Once the content of the page (video is loaded) is ready, we grab
values from the page and instantiate the moment-player (which is currently
all the buttons and the ground-truth slider)
*/

document.addEventListener("DOMContentLoaded", function () {

    let [predictions, durations, video_names] = grab_info();

    for (let i = 0; i < predictions.length; i++) {
        videojs('player_'+(i+1)).ready(function () {

            // event triggered when reaching SliderGT-1
            this.on('ended', function () {
                if (loop == true) {
                    const t_min = Math.min(...slider_values("SliderPred_"+this.id_.split("_")[1]));
                    this.currentTime(t_min);
                    this.play();
                };
            });

            this.on('timeupdate', function () {
                let t_max = Math.max(...slider_values("SliderPred_"+this.id_.split("_")[1]));
                if (this.currentTime() > t_max) {
                    this.pause();
                    this.trigger('ended');
                }
                let slider_i = document.getElementById("SliderPred_"+this.id_.split("_")[1]);
                let triad    = slider_i.noUiSlider.get();
                triad[1]     = this.currentTime().toFixed(1);
                slider_i.noUiSlider.set(triad);
            });


            this.on('loadedmetadata', function () {
                let video_duration = this.duration();
                const index = parseInt(this.id_.split("_")[1]);
                let video_duration_pred = durations[video_names[index-1]];

                document.getElementById("prediction-show-"+index).innerHTML = "Video ID: ".bold() + video_names[index-1] + 
                                        " --" + " Interval: ".bold() + predictions[index-1][0].toFixed(2) + " - "
                                        + predictions[index-1][1].toFixed(2) + " --" + " Rank: ".bold() + (index);

                if (video_duration < predictions[index-1][1]){
                        predictions[index-1][1] = video_duration;
                }
                if (video_duration_pred < predictions[index-1][0]){
                        predictions[index-1][0] = 0;
                }

                create_html_slider(parent_id    = "Predictions_" + index, 
                                    element_id  = `SliderPred_${index}`, 
                                    pair        = predictions[index-1],
                                    max_value   = Math.max(video_duration,predictions[index-1][1]), 
                                    step        = 0.01, 
                                    onchange    = undefined);

                this.currentTime(predictions[index-1][0]);
            });
        });
    };
});