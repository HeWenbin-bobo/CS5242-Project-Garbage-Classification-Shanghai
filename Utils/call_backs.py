from bokeh.models import CustomJS

# handle the currently selected article
def selected_code():
    code = """
            var truths = [];
            var predictions = [];
            cb_data.source.selected.indices.forEach(index => truths.push(source.data['truths'][index]));
            cb_data.source.selected.indices.forEach(index => predictions.push(source.data['predictions'][index]));
            const truth = "<h4>Truth label: " + truths[0].toString().replace(/<br>/g, ' ') + "</h4>";
            const prediction = "<p1><b>prediction:</b> " + predictions[0].toString().replace(/<br>/g, ' ') + "</p1><br>";
            current_selection.text = truth + prediction;
            current_selection.change.emit();
    """
    return code
            # const truth = "<h4>Truth label: " + truths[-1].toString().replace(/<br>/g, ' ') + "</h4>";
            # const prediction = "<p1><b>prediction:</b> " + predictions[-1].toString().replace(/<br>/g, ' ') + "</p1><br>";
            # current_selection.text = truth + prediction;
            # console.log(5 + 6);
# handle the keywords and search
def input_callback(plot, source, out_text, keywords, classes_num): 

    # slider call back for cluster selection
    callback = CustomJS(args=dict(p=plot, source=source, out_text=out_text, keywords=keywords, classes_num=classes_num), code="""
                var key = text.value;
				key = key.toLowerCase();
				var class_index = slider.value;
                var data = source.data; 
                
                var x = data['x'];
                var y = data['y'];
                const x_backup = data['x_backup'];
                const y_backup = data['y_backup'];
                const indexes = data['indexes'];
                const labels = data['labels'];
                const truths = data['truths'];
                const predictions = data['predictions'];
                var predictions_lowerCase = [];
                predictions.forEach(prediction => predictions_lowerCase.push(prediction.toLowerCase()));

                if (class_index == classes_num.toString()) {
                    out_text.text = 'Keywords: Slide to specific class to see the keywords.';
                    for (var i = 0; i < x.length; i++) {
						if(predictions_lowerCase[i].includes(key)) {
							x[i] = x_backup[i];
							y[i] = y_backup[i];
						} else {
							x[i] = undefined;
							y[i] = undefined;
						}
                    }
                }
                else {
                    out_text.text = 'Keywords: ' + keywords[Number(class_index)];
                    for (var i = 0; i < x.length; i++) {
                        if(indexes[i] == Number(class_index)) {
                            console.log(predictions[i])
							if(predictions_lowerCase[i].includes(key)) {
								x[i] = x_backup[i];
								y[i] = y_backup[i];
							} else {
								x[i] = undefined;
								y[i] = undefined;
							}
                        } else {
                            x[i] = undefined;
                            y[i] = undefined;
                        }
                    }
                }

            source.change.emit();
            """)
    return callback