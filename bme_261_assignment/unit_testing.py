# from database import Database
# from util import Util
# from main import process_state

# database_controller = Database()
# util_controller = Util(database_controller)


# def test_update_get_rectangle_polygon_contour_data():
#     """
#     Testing our Database class to ensure that data does get successfully updated. Once updated,
#     the data is retreieved again and compared to the original data provided in the update operation
#     """
#     data = {
#         "area_threshold": 0,
#         "lower_length_threshold": 4,
#         "upper_length_threshold": 15,
#     }
#     database_controller.update_rectangle_polygon_contour_data(data)
#     config_data = database_controller.data["rectangle_polygon_contour"]
#     for key in data:
#         assert data[key] == config_data[key], f"Unexpected value inequality for {key}"


# def test_process_state():

#     """
#     Testing to ensure that the process_state function returns correct values for flow rate representation based on test values
#     for top border, bottom border, and various ball positions which lead to all the different flow rate states
#     """

#     test_top_border, test_bottom_border = 100, 500
#     ball_position = {
#         "not_found": -1,
#         "normal_flow_rate": 300,
#         "flow_rate_too_high": 50,
#         "flow_rate_too_low": 550,
#     }
#     flow_rate_state = {
#         "not_found": -2,
#         "normal_flow_rate": 0,
#         "flow_rate_too_high": 1,
#         "flow_rate_too_low": -1,
#     }

#     assert (
#         int(
#             (
#                 process_state(
#                     test_top_border, test_bottom_border, ball_position["not_found"]
#                 )
#             )[0]
#         )
#         == flow_rate_state["not_found"]
#     ), "Process state failed to catch ball not found state"
#     assert (
#         int(
#             (
#                 process_state(
#                     test_top_border,
#                     test_bottom_border,
#                     ball_position["normal_flow_rate"],
#                 )
#             )[0]
#         )
#         == flow_rate_state["normal_flow_rate"]
#     ), "Process state failed to catch normal flow rate state"
#     assert (
#         int(
#             (
#                 process_state(
#                     test_top_border,
#                     test_bottom_border,
#                     ball_position["flow_rate_too_high"],
#                 )
#             )[0]
#         )
#         == flow_rate_state["flow_rate_too_high"]
#     ), "Process state failed to catch flow rate too high state"
#     assert (
#         int(
#             (
#                 process_state(
#                     test_top_border,
#                     test_bottom_border,
#                     ball_position["flow_rate_too_low"],
#                 )
#             )[0]
#         )
#         == flow_rate_state["flow_rate_too_low"]
#     ), "Process state failed to catch flow rate too low state"
#     pass


# def test_get_color_image():
#     """
#     Testing to see that the Util class function, get_color_image(), returns the correct shape. For colored images,
#     the shape would have a length of 3. These are rows, columns, and channels (RGB/BGR)
#     """
#     colored_image = util_controller.get_color_image()
#     assert len(colored_image.shape) == 3
#     pass
