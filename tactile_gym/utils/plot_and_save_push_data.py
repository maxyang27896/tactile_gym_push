import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os, sys, shutil


def plot_and_save_push_plots(env, data, data_columns, trials, directory, result_type):

    # create dataframe
    df = pd.DataFrame(data, columns = data_columns)
    pd.DataFrame(data).to_csv(os.path.join(directory, "{}_result.csv".format(result_type)), header=data_columns)

    loss_contact = False
    for trial in range(trials):
        fig_xy, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df.query("trial==@trial")["tcp_x"], df.query("trial==@trial")["tcp_y"], "bs", label='tcp psosition')
        ax.plot(df.query("trial==@trial").query("contact==@loss_contact")["tcp_x"], df.query("trial==@trial").query("contact==@loss_contact")["tcp_y"], "g+", markersize=20)
        ax.plot(df.query("trial==@trial")["contact_x"], df.query("trial==@trial")["contact_y"], "rs", label='contact psosition')
        ax.plot(df.query("trial==@trial").query("contact==@loss_contact")["contact_x"], df.query("trial==@trial").query("contact==@loss_contact")["contact_y"], "gx", markersize=20)
        ax.plot(df.query("trial==@trial").query("goal_reached==True")['goal_x'], df.query("trial==@trial").query("goal_reached==True")['goal_y'], "x", markersize=20, markeredgewidth=3, markeredgecolor="green", label="reached goal")
        
        # If last row is goal is not reached, plot last goal
        if not df.query("trial==@trial")["goal_reached"].iloc[-1]:
            ax.plot(df.query("trial==@trial")['goal_x'].iloc[-1], df.query("trial==@trial")['goal_y'].iloc[-1], "x", markersize=20, markeredgewidth=3, markeredgecolor="black", label=" not reached goal")


        # Plot orn arrows
        for i, rows in df.query("trial==@trial").iterrows():
            if i % 10 == 0:
                tcp_x, tcp_y, tcp_Rz = rows["tcp_x"], rows["tcp_y"], rows["tcp_Rz"]
                tcp_dx, tcp_dy = 0.05 * np.cos(tcp_Rz), 0.05 * np.sin(tcp_Rz)
                tcp_x, tcp_y = tcp_x + tcp_dx*0.2, tcp_y + tcp_dy*0.2
                plt.arrow(tcp_x, tcp_y, tcp_dx, tcp_dy, color='b')
                obj_x, obj_y, obj_Rz= rows["contact_x"], rows["contact_y"], rows["contact_Rz"]
                obj_dx, obj_dy = 0.05 * np.cos(obj_Rz), 0.05 * np.sin(obj_Rz)
                obj_x, obj_y = obj_x + obj_dx*0.2, obj_y + obj_dy*0.2
                plt.arrow(obj_x, obj_y, obj_dx, obj_dy, color='r')
        
        
        ax.set_xlabel("x workframe")
        ax.set_ylabel("y workframe")
        ax.set_xlim([env.robot.arm.TCP_lims[0, 0], env.robot.arm.TCP_lims[0, 1]])
        ax.set_ylim([env.robot.arm.TCP_lims[1, 0], env.robot.arm.TCP_lims[1, 1]])
        ax.legend()
        fig_xy.savefig(os.path.join(directory, "workframe_plot_trial_{}.png".format(trial)))
        plt.close(fig_xy)

        fig_time_xy, axs = plt.subplots(3, 2, figsize=(14, 7.5), gridspec_kw={"width_ratios": [1, 1]})
        axs[0, 0].plot(df.query("trial==@trial")["time_steps"], df.query("trial==@trial")["tcp_x"], "bs", label='tcp ')
        axs[0, 0].plot(df.query("trial==@trial").query("contact==@loss_contact")["time_steps"], df.query("trial==@trial").query("contact==@loss_contact")["tcp_x"], "g+", markersize=20)
        axs[0, 0].plot(df.query("trial==@trial")["time_steps"], df.query("trial==@trial")["contact_x"], "rs", label='contact')
        axs[0, 0].plot(df.query("trial==@trial").query("contact==@loss_contact")["time_steps"], df.query("trial==@trial").query("contact==@loss_contact")["contact_x"], "gx", markersize=20)
        axs[0, 0].axhline(y=df.query("trial==@trial")["goal_x"].iloc[0], color='g', linestyle='-', linewidth=2)
        axs[0, 0].set_xlabel("Time steps (s)")
        axs[0, 0].set_ylabel("x axis workframe")
        axs[0, 0].set_ylim([env.robot.arm.TCP_lims[0, 0], env.robot.arm.TCP_lims[0, 1]])
        axs[0, 0].legend()
        axs[0, 0].grid()

        axs[0, 1].plot(df.query("trial==@trial")["time_steps"], df.query("trial==@trial")["tcp_y"], "bs", label='tcp')
        axs[0, 1].plot(df.query("trial==@trial").query("contact==@loss_contact")["time_steps"], df.query("trial==@trial").query("contact==@loss_contact")["tcp_y"], "g+", markersize=20)
        axs[0, 1].plot(df.query("trial==@trial")["time_steps"], df.query("trial==@trial")["contact_y"], "rs", label='contact')
        axs[0, 1].plot(df.query("trial==@trial").query("contact==@loss_contact")["time_steps"], df.query("trial==@trial").query("contact==@loss_contact")["contact_y"], "gx", markersize=20)
        axs[0, 1].axhline(y=df.query("trial==@trial")["goal_y"].iloc[0], color='g', linestyle='-', linewidth=2)
        axs[0, 1].set_xlabel("Time steps (s)")
        axs[0, 1].set_ylabel("y axis workframe")
        axs[0, 1].set_ylim([env.robot.arm.TCP_lims[1, 0], env.robot.arm.TCP_lims[1, 1]])
        axs[0, 1].legend()
        axs[0, 1].grid()

        axs[1, 0].plot(df.query("trial==@trial")["time_steps"], abs(df.query("trial==@trial")["goal_Rz"] - df.query("trial==@trial")["contact_Rz"]), "gs", label='goal Rz')
        axs[1, 0].axhline(y=0, color='b', linestyle='-', linewidth=2)
        axs[1, 0].set_xlabel("Time steps (s)")
        axs[1, 0].set_ylabel("contact Rz goal workframe")
        axs[1, 0].set_ylim(-np.deg2rad(45), np.deg2rad(135))
        axs[1, 0].grid()

        axs[1, 1].plot(df.query("trial==@trial")["time_steps"], abs(df.query("trial==@trial")["contact_Rz"] - df.query("trial==@trial")["tcp_Rz"]), "bs", label='goal Rz')
        axs[1, 1].axhline(y=0, color='g', linestyle='-', linewidth=2)
        axs[1, 1].set_xlabel("Time steps (s)")
        axs[1, 1].set_ylabel("contact Rz tcp workframe")
        axs[1, 1].set_ylim(-np.deg2rad(10), np.deg2rad(30))
        axs[1, 1].grid()

        axs[2, 0].plot(df.query("trial==@trial")["time_steps"], df.query("trial==@trial")["action_y"], "r-", label='action_y')
        axs[2, 0].set_xlabel("Time steps (s)")
        axs[2, 0].set_ylabel("action y")
        axs[2, 0].set_ylim(-0.3, 0.3)
        axs[2, 0].grid()

        axs[2, 1].plot(df.query("trial==@trial")["time_steps"], df.query("trial==@trial")["action_Rz"], "r-", label='action Rz')
        axs[2, 1].set_xlabel("Time steps (s)")
        axs[2, 1].set_ylabel("action Rz")
        axs[2, 1].set_ylim(-0.3, 0.3)
        axs[2, 1].grid()
        
        fig_time_xy.savefig(os.path.join(directory, "time_plot_trial_{}.png".format(trial)))
        plt.close(fig_time_xy)


def plot_and_save_training(env, data, trial, data_columns, directory):

    # plot and save results
    if not os.path.exists(directory):
        os.mkdir(directory)
        df = pd.DataFrame(data, columns = data_columns)
        pd.DataFrame(data).to_csv(os.path.join(directory, "{}_result.csv".format("training")))
    else:
        df = pd.DataFrame(data, columns = data_columns)
        pd.DataFrame(data).to_csv(os.path.join(directory, "{}_result.csv".format("training")), mode='a', header=False)

    loss_contact = False
    fig_xy, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df["tcp_x"], df["tcp_y"], "bs", label='tcp psosition')
    ax.plot(df.query("contact==@loss_contact")["tcp_x"], df.query("contact==@loss_contact")["tcp_y"], "g+", markersize=20)
    ax.plot(df["contact_x"], df["contact_y"], "rs", label='contact psosition')
    ax.plot(df.query("contact==@loss_contact")["contact_x"], df.query("contact==@loss_contact")["contact_y"], "gx", markersize=20)
    ax.plot(df["goal_x"].iloc[0], df["goal_y"].iloc[0], "x", markersize=20, markeredgecolor="black", label="goal position")
    
    # Plot orn arrows
    for i, rows in df.iterrows():
        if i % 10 == 0:
            tcp_x, tcp_y, tcp_Rz = rows["tcp_x"], rows["tcp_y"], rows["tcp_Rz"]
            tcp_dx, tcp_dy = 0.05 * np.cos(tcp_Rz), 0.05 * np.sin(tcp_Rz)
            tcp_x, tcp_y = tcp_x + tcp_dx*0.2, tcp_y + tcp_dy*0.2
            plt.arrow(tcp_x, tcp_y, tcp_dx, tcp_dy, color='b')
            obj_x, obj_y, obj_Rz= rows["contact_x"], rows["contact_y"], rows["contact_Rz"]
            obj_dx, obj_dy = 0.05 * np.cos(obj_Rz), 0.05 * np.sin(obj_Rz)
            obj_x, obj_y = obj_x + obj_dx*0.2, obj_y + obj_dy*0.2
            plt.arrow(obj_x, obj_y, obj_dx, obj_dy, color='r')
    
    
    ax.set_xlabel("x workframe")
    ax.set_ylabel("y workframe")
    ax.set_xlim([env.robot.arm.TCP_lims[0, 0], env.robot.arm.TCP_lims[0, 1]])
    ax.set_ylim([env.robot.arm.TCP_lims[1, 0], env.robot.arm.TCP_lims[1, 1]])
    ax.legend()
    fig_xy.savefig(os.path.join(directory, "workframe_plot_trial_{}.png".format(trial)))
    plt.close(fig_xy)

    # fig_time_xy, axs = plt.subplots(1, 2, figsize=(14, 3.75), gridspec_kw={"width_ratios": [1, 1]})
    # axs[0].plot(df["time_steps"], df["tcp_x"], "bs", label='tcp ')
    # axs[0].plot(df.query("contact==@loss_contact")["time_steps"], df.query("contact==@loss_contact")["tcp_x"], "g+", markersize=20)
    # axs[0].plot(df["time_steps"], df["contact_x"], "rs", label='contact')
    # axs[0].plot(df.query("contact==@loss_contact")["time_steps"], df.query("contact==@loss_contact")["contact_x"], "gx", markersize=20)
    # axs[0].set_xlabel("Time steps (s)")
    # axs[0].set_ylabel("x axis workframe")
    # axs[0].set_ylim([env.robot.arm.TCP_lims[0, 0], env.robot.arm.TCP_lims[0, 1]])
    # axs[0].legend()
    # axs[1].plot(df["time_steps"], df["tcp_y"], "bs", label='tcp')
    # axs[1].plot(df.query("contact==@loss_contact")["time_steps"], df.query("contact==@loss_contact")["tcp_y"], "g+", markersize=20)
    # axs[1].plot(df["time_steps"], df["contact_y"], "rs", label='contact')
    # axs[1].plot(df.query("contact==@loss_contact")["time_steps"], df.query("contact==@loss_contact")["contact_y"], "gx", markersize=20)
    # axs[1].set_xlabel("Time steps (s)")
    # axs[1].set_ylabel("y axis workframe")
    # axs[1].set_ylim([env.robot.arm.TCP_lims[1, 0], env.robot.arm.TCP_lims[1, 1]])
    # axs[1].legend()
    # fig_time_xy.savefig(os.path.join(directory, "time_plot_trial_{}.png".format(trial)))
    # plt.close(fig_time_xy)