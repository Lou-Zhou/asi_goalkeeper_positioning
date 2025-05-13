"""Data visualisation - legacy from un-xPass code"""
import pandas as pd
from mplsoccer import Pitch
import numpy as np

def plot_action(
    action: pd.Series,
    selection = None,
    title = None,
    surface=None,
    show_action=True,
    show_visible_area=True,
    ax=None,
    surface_kwargs={},
) -> None:
    """Plot a SPADL action with 360 freeze frame.

    Parameters
    ----------
    action : pandas.Series
        A row from the actions DataFrame.
    surface : np.arry, optional
        A surface to visualize on top of the pitch.
    ax : matplotlib.axes.Axes, optional
        The axes to plot on.
    surface_kwargs : dict, optional
        Keyword arguments to pass to the surface plotting function.
    """
    # parse freeze frame
    freeze_frame = pd.DataFrame.from_records(action["freeze_frame_360"])
    visible_area = action["visible_area_360"]
    teammate_locs = freeze_frame[freeze_frame.teammate]
    opponent_locs = freeze_frame[~freeze_frame.teammate]
    #teammate_locs = pd.merge(teammate_locs, selection, left_on = ["destination_x", "destination_y"], right_on = ["x", "y"])
    #return teammate_locs
    # set up pitch
    p = Pitch(pitch_type="custom", pitch_length=105, pitch_width=68)
    if ax is None:
        _, ax = p.draw(figsize=(12, 8))
    else:
        p.draw(ax=ax)

    # plot action
    if show_action:
        p.arrows(
            action["start_x"],
            action["start_y"],
            action["end_x"],
            action["end_y"],
            color="black",
            headwidth=5,
            headlength=5,
            width=1,
            ax=ax,
        )
    # plot visible area
    if show_visible_area:
        p.polygon([visible_area], color=(236 / 256, 236 / 256, 236 / 256, 0.5), ax=ax)
    # plot freeze frame
    p.scatter(opponent_locs.x, opponent_locs.y, c="r", s=30, ax=ax, marker = "x")
    p.scatter(teammate_locs.x, teammate_locs.y, c="b", s=30, ax=ax, marker = "o")
   #p.scatter(teammate_locs.x, teammate_locs.y, s=30, ax=ax, facecolors=teammate_locs.selection_probability, edgecolors=teammate_locs.selection_probability)
    p.scatter(action["start_x"], action["start_y"], c="w", ec = "k", s=20, ax=ax)
    #p.scatter(60, 40, c="w", ec = "k", s=20, ax=ax)
    if title:
        ax.set_title(title, fontsize=16)
    # plot surface
    if surface is not None:
        ax.imshow(surface, extent=[0.0, 105.0, 0.0, 68.0], origin="lower", **surface_kwargs)

    return ax
def plot_action_og(
    action: pd.Series,
    surface=None,
    show_action=False,
    show_visible_area=True,
    ax=None,
    log=  False,
    title = None,
    surface_kwargs={},
) -> None:
    """Plot a SPADL action with 360 freeze frame.

    Parameters
    ----------
    action : pandas.Series
        A row from the actions DataFrame.
    surface : np.arry, optional
        A surface to visualize on top of the pitch.
    ax : matplotlib.axes.Axes, optional
        The axes to plot on.
    surface_kwargs : dict, optional
        Keyword arguments to pass to the surface plotting function.
    """
    # parse freeze frame
    #print("AAA")
    freeze_frame = pd.DataFrame.from_records(action["freeze_frame_360"])
    visible_area = action["visible_area_360"]
    teammate_locs = freeze_frame[freeze_frame.teammate]
    opponent_locs = freeze_frame[~freeze_frame.teammate]

    # set up pitch
    p = Pitch(pitch_type="custom", pitch_length=105, pitch_width=68)
    if title:
        ax.set_title(title, fontsize=16)
    if ax is None:
        _, ax = p.draw(figsize=(12, 8))
    else:
        p.draw(ax=ax)

    # plot action
    if show_action:
        p.arrows(
            action["start_x"],
            action["start_y"],
            action["end_x"],
            action["end_y"],
            color="black",
            headwidth=5,
            headlength=5,
            width=1,
            ax=ax,
        )
    # plot visible area
    if show_visible_area:
        p.polygon([visible_area], color=(236 / 256, 236 / 256, 236 / 256, 0.5), ax=ax)
    # plot freeze frame
    p.scatter(opponent_locs.x, opponent_locs.y, c="r", s=30, ax=ax, marker = "x")
    p.scatter(teammate_locs.x, teammate_locs.y, c="b", s=30, ax=ax, marker = "o")
    p.scatter(action["start_x"], action["start_y"], c="w", ec = "k", s=20, ax=ax)

    # plot surface
    if surface is not None:
        if log:
            surface = np.log(surface)
        ax.imshow(surface, extent=[0.0, 105.0, 0.0, 68.0], origin="lower", **surface_kwargs)

    return ax