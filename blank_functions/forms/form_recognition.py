from blank_functions.forms.forms import Form

class FormRecognition:

    def __init__(self, image, template_path, json_path, answers, version):
        self.image = image
        self.template_path = template_path
        self.json_path = json_path
        self.answers = answers
        self.version = version

    def __repr__(self):
        return (f"FormRecognition(template_path={self.template_path}, "
                f"json_path={self.json_path}, answers={self.answers}, "
                f"version={self.version})")

    def run_pipeline(self):
        form = Form(self.version)   
        form.load_meta_from_json(self.json_path)  # Load meta data from template json
        form.load_image(self.image)  # Load user image
        form.load_template(self.template_path)  # Load template
        form.align_form(scale_factor=1.0)  # Align user image to template
        form.recalculate_cells()
        form.style_image()
        # form.remove_cells_lines()  # Remove cells lines
        # form.load_correct_answers(self.answers)  # Load correct answers from answers excel
        # form.get_symbals_from_image()  # Get symbols from image
        # form.get_rows_contour()  # Get rows contour

        return form