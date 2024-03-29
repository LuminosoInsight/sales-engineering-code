
  $(function() {

    var submit_form = function(e) {
      $.getJSON($SCRIPT_ROOT + '/_step1', {
        token: $('input[name=token]').val()
      }, function(data) {
        //$('#result').text(data.result);
        //console.log("server response:" + JSON.stringify(data));
        w = JSON.stringify(data);
        //$('#result').text(w);

        var obj = JSON.parse(w);
        var d = {};
        var selectSurveyCode = '<br><h4>Step 2: Select which survey you would like to upload:</h4><p>';
        selectSurveyCode += '<select class="form-control" name="surveys">';
        
        for(var key in obj){
          if (obj.hasOwnProperty(key)){
           var value=obj[key];
           console.log("key:" + key);
           console.log("value:" + value);
           d[key] = value;
           selectSurveyCode += '<option value='+value+'>'+key+'</option>';
          }
        }
        selectSurveyCode += '</select><br>'
        selectSurveyCode += '<input type="submit" class="btn btn-default btn-sm" name="submitSurveySelect" value="Select this Survey"/>'

        $('#surveys').html(selectSurveyCode);

        $('input[name=submitSurveySelect]').bind('click', function() {
          console.log("sending over: " + $('select[name=surveys]').val() + " and token: " + $('input[name=token]').val());
         $.getJSON($SCRIPT_ROOT + '/_step2', {
          //sending over the sid of the survey
          sid: $('select[name=surveys]').val(), 
          token: $('input[name=token]').val()
          }, function(data) {

            questions = JSON.stringify(data);
            var apple = JSON.parse(questions);

            for(var key in apple){ //Create selections for Documents
              if (apple.hasOwnProperty(key)){
                 var value=apple[key];

                 if (key == 'text'){
                  var selectTextCode = '<br><h4>Step 3: Select which questions to have as documents:</h4><p>';
                  selectTextCode += '<table class="table table-striped"><tr><th>Select</th><th>Question ID</th><th>Question</th></tr>';
                  for (var i = 0; i < apple.text.length; i++){
                     selectTextCode += '<tr>';
                     selectTextCode += '<td><input type="checkbox" name="docSelect[]" value="'+apple.text[i][0]+'"/></td>';
                     selectTextCode += '<td>' + apple.text[i][0] + '</td><td>' + apple.text[i][1] + '</td></tr>';
                     console.log("this value is: " + apple.text[i][0]);
                  } 
                  selectTextCode += '</table>';
                  
                 } 

                 if (key == 'subsets'){ //Create selections for Subsets
                  var selectSubsetCode = '<br><h4>Step 4: Select which questions to have as subsets:</h4><br>';
                  selectSubsetCode += '<table class="table table-striped"><tr><th>Select</th><th>Question ID</th><th>Question</th></tr>';
                  for (var i = 0; i < apple.subsets.length; i++){
                     selectSubsetCode += '<tr>';
                     selectSubsetCode += '<td><input type="checkbox" class="checkbox" name="subSelect[]" value="'+apple.subsets[i][0]+'"/></td>';
                     selectSubsetCode += '<td>' + apple.subsets[i][0] + '</td><td>' + apple.subsets[i][1] + '</td></tr>';
                  }
                  selectSubsetCode += '</table>';
                  selectSubsetCode += '<br><h4>Step 5: Click below to create a new project.</h4>A project url will appear below once the project is completed.<br><br><input type="submit" class="btn btn-primary" name="formSubmit" value="Create Project"/>';
                  selectSubsetCode += ' <img id="loading" padding-left="50px" alt="spinning wheel" src={{ url_for('static', filename = 'images/ajax-loader.gif') }}  style="display:none"><br>';
                 }
                 
              }
             }
            $('#text').html(selectTextCode);
            $('#subsets').html(selectSubsetCode);
            //$('#questions').text(questions);


              $('input[name=formSubmit]').bind('click', function() {
                if ($('input[name="docSelect[]"]:checked').length < 1) {
                  alert("You must select at least one text field to be analyzed!")
                  return;
                }
                $('input[name=formSubmit]').attr("disabled", true);
                console.log("button disabled")
                $("#loading").show();
                $('#uploaded').html("");
                console.log("load bar started");
                var sub_checked = [];   
                $('input[name="subSelect[]"]:checked').each(function() {
                    sub_checked.push($(this).val());
                  });
                var sub_checkedStr = JSON.stringify(sub_checked);
                var doc_checked = [];   
                $('input[name="docSelect[]"]:checked').each(function() {
                    doc_checked.push($(this).val());
                  });
                var doc_checkedStr = JSON.stringify(doc_checked);
                console.log("subset: " + sub_checkedStr);
                console.log("subset: " + doc_checkedStr);
                console.log("title: " +  $('#surveys option:selected').text());
                $.getJSON($SCRIPT_ROOT + '/_step3', {
               //sending over the sid of the survey
                 sid: $('select[name=surveys]').val(), 
                 //add proj_name: 
                 title: $('#surveys option:selected').text(),
                 token: $('input[name=token]').val(),
                 text_qs: doc_checkedStr,
                 subset_qs: sub_checkedStr
                }, function(data) {
                    projurl = JSON.stringify(data);
                    var x= JSON.parse(projurl);
                    console.log("url: " +x.url);
                    $("#loading").hide();
                    $("#status").hide();
                    console.log("spinner removed");
                    var uploadedCode = "<br><h4>Your project has finished calculating. ";
                    uploadedCode += "<a target='_blank' href="+x.url+">Click here</a> to view it.</h4><br><br>To create a new project, please refresh this page.<br><br>";
                    $('#uploaded').html(uploadedCode);
                    console.log("done");
                });
                // These statuses are just for demo purposes. They do not reflect reality.
                console.log("data submitted");
                $("#status1").show();
                 setTimeout(function (){

                    // Something you want delayed.
                    $("#status2").show();

                  }, 1000);
                 setTimeout(function (){

                    // Something you want delayed.
                     $("#status3").show();

                  }, 2500);
                 setTimeout(function (){

                    // Something you want delayed.
                   $("#status4").show();

                  }, 3500);
              });

                 
           });  //end of step2 function(data)
         });



        }); //end of step1 function(data)
        return false;
    }; //end of variable submit_form

    $('input[name=sub_cred]').bind('click', submit_form);


    
    $('input[type=text]').bind('keydown', function(e) {
      if (e.keyCode == 13) {
        submit_form(e);
      }
    });

    



    
    



  }); //end of main function


