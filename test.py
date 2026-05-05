from PEMOP import ContextualMultiObjectiveFunction, VAEEnhancedCMOBO, SimpleDiffusionContextualMOBO
import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import qmc
import os
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description='Run DTLZ optimization experiments (VAE-CMOBO / DDPM-CMOBO)')
    parser.add_argument('--acquisition_type', type=str, default='UCB', choices=['UCB', 'TS'],
                        help='Acquisition function type (default: UCB)')
    parser.add_argument('--problem', type=str, default='dtlz2',
                        choices=['dtlz1', 'dtlz2', 'dtlz3'],
                        help='DTLZ problem to optimize (default: dtlz2)')
    parser.add_argument('--n_runs', type=int, default=1, help='Number of optimization runs')
    parser.add_argument('--n_iter', type=int, default=5, help='Number of iterations per run')
    parser.add_argument('--n_objectives', type=int, default=2, help='Number of objectives')
    parser.add_argument('--n_variables', type=int, default=5, help='Number of variables')
    parser.add_argument('--beta', type=float, default=1.0, help='Beta parameter')
    parser.add_argument('--model_type', type=str, default='ExactGP', help='GP model type')
    parser.add_argument('--method_name', type=str, default='VAE-CMOBO',
                        choices=['VAE-CMOBO', 'DDPM-CMOBO'],
                        help='Method to run')
    parser.add_argument('--m_tasks', type=int, default=8, help='Number of tasks/contexts')
    parser.add_argument('--m_samples', type=int, default=20, help='Number of samples per task')
    parser.add_argument('--genai_training_frequency', type=int, default=None,
                        help='Training frequency for GenAI models. Default: 2 for VAE, 1 for DDPM')
    parser.add_argument('--genai_top_p', type=float, default=None,
                        help='Top percentage (0-1) for GenAI training data. Default: 0.1')
    parser.add_argument('--genai_num_candidates', type=int, default=None,
                        help='Number of candidates from GenAI. Default: 30000 for VAE, 15000 for DDPM')
    return parser.parse_args()


def generate_and_save_contexts(n_contexts, context_dim, file_path):
    sampler = qmc.LatinHypercube(d=context_dim)
    contexts = sampler.random(n=n_contexts)
    contexts_tensor = torch.tensor(contexts, dtype=torch.float32)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    torch.save(contexts_tensor, file_path)
    return contexts_tensor


def plot_contexts(contexts):
    if contexts.shape[1] == 2:
        plt.figure(figsize=(8, 8))
        plt.scatter(contexts[:, 0], contexts[:, 1])
        plt.xlabel('Context Dimension 1')
        plt.ylabel('Context Dimension 2')
        plt.title('Distribution of Contexts')
        plt.savefig('context_distribution.png')
        plt.close()
    else:
        print("Cannot visualize contexts with more than 2 dimensions in 2D")


def vae_optimization_loop_test(problem_name='dtlz2', n_runs=1, n_iter=5, n_objectives=2,
                                n_variables=5, temp_beta=1.0, model_type='ExactGP', m_tasks=8,
                                m_samples=20, acquisition_type='UCB', genai_training_frequency=2,
                                genai_top_p=0.1, genai_num_candidates=30000):
    obj_func = ContextualMultiObjectiveFunction(func_name=problem_name,
                                                n_objectives=n_objectives,
                                                n_variables=n_variables)
    directory_path = f'result/{problem_name}'
    os.makedirs(directory_path, exist_ok=True)

    n_contexts = m_tasks
    contexts_file = 'data/context_{}_{}.pth'.format(n_contexts, obj_func.context_dim)
    if os.path.exists(contexts_file):
        contexts = torch.load(contexts_file)
    else:
        contexts = generate_and_save_contexts(n_contexts, obj_func.context_dim, contexts_file)
    plot_contexts(contexts)

    timestamp = "{}_{}_{}_{}_{}_hv".format(problem_name, n_variables, n_objectives,
                                           model_type, acquisition_type)

    for run in range(n_runs):
        print(f"Starting run {run + 1}/{n_runs}")

        optimizer = VAEEnhancedCMOBO(
            objective_func=obj_func,
            true_conditional=False,
            model_type=model_type,
            problem_name=problem_name,
            acquisition_type=acquisition_type,
            vae_training_frequency=genai_training_frequency,
            top_p=genai_top_p,
            vae_num_candidates=genai_num_candidates
        )

        n_initial_points = m_samples
        X_init = torch.zeros(n_initial_points * n_contexts, obj_func.input_dim + obj_func.context_dim)
        for i in range(n_contexts):
            start_idx = i * n_initial_points
            end_idx = (i + 1) * n_initial_points
            init_file = 'data/init_points_context_{}_{}_{}.pth'.format(i, obj_func.input_dim, n_initial_points)
            if os.path.exists(init_file):
                init_points = torch.load(init_file)
            else:
                sampler = qmc.LatinHypercube(d=obj_func.input_dim)
                init_points = torch.tensor(sampler.random(n=n_initial_points), dtype=torch.float32)
                os.makedirs('data', exist_ok=True)
                torch.save(init_points, init_file)
            X_init[start_idx:end_idx, :obj_func.input_dim] = init_points
            X_init[start_idx:end_idx, obj_func.input_dim:] = contexts[i].repeat(n_initial_points, 1)

        Y_init = obj_func.evaluate(X_init)
        X_opt, Y_opt = optimizer.optimize(X_init, Y_init, contexts, n_iter=n_iter, run=run)

        run_data = {}
        for i, context in enumerate(contexts):
            context_key = tuple(context.numpy())
            if context_key in optimizer.context_pareto_fronts:
                run_data[context_key] = {
                    'pareto_set_history': optimizer.context_pareto_sets[context_key],
                    'pareto_front_history': optimizer.context_pareto_fronts[context_key],
                    'hv_history': optimizer.context_hv[context_key]
                }
                print(f"Run {run + 1}, Context {i}: Final HV = {optimizer.context_hv[context_key][-1]:.4f}")
            else:
                print(f"Run {run + 1}, Context {i}: No Pareto front found")

        save_path = f'result/{problem_name}/VAE-CMOBO_{genai_training_frequency}_{genai_top_p}_{timestamp}_run_{run}.pth'
        torch.save(run_data, save_path)
        print(f"Run {run + 1} saved to {save_path}")

        fig, axes = plt.subplots(4, 4, figsize=(20, 20))
        fig.suptitle(f'VAE-CMOBO: HV History (Run {run + 1})', fontsize=16)
        for i, context in enumerate(contexts):
            row, col = i // 4, i % 4
            ax = axes[row, col]
            context_key = tuple(context.numpy())
            if context_key in run_data:
                ax.plot(range(len(run_data[context_key]['hv_history'])),
                        run_data[context_key]['hv_history'])
            ax.set_title(f'Context {i}')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Hypervolume')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f'result/{problem_name}/VAE-CMOBO_{genai_training_frequency}_{genai_top_p}_{timestamp}_run_{run}.png')
        plt.close()


def simple_ddpm_optimization_loop_test(problem_name='dtlz2', n_runs=1, n_iter=5, n_objectives=2,
                                        n_variables=5, temp_beta=1.0, model_type='ExactGP', m_tasks=8,
                                        m_samples=20, acquisition_type='UCB', genai_training_frequency=1,
                                        genai_top_p=0.1, use_batch_norm=False, genai_num_candidates=15000):
    obj_func = ContextualMultiObjectiveFunction(func_name=problem_name,
                                                n_objectives=n_objectives,
                                                n_variables=n_variables)
    directory_path = f'result/{problem_name}'
    os.makedirs(directory_path, exist_ok=True)

    n_contexts = m_tasks
    contexts_file = 'data/context_{}_{}.pth'.format(n_contexts, obj_func.context_dim)
    if os.path.exists(contexts_file):
        contexts = torch.load(contexts_file)
    else:
        contexts = generate_and_save_contexts(n_contexts, obj_func.context_dim, contexts_file)
    plot_contexts(contexts)

    timestamp = "{}_{}_{}_{}_{}_hv".format(problem_name, n_variables, n_objectives,
                                           model_type, acquisition_type)

    print(f"Starting DDPM-CMOBO: {n_runs} runs, {n_contexts} contexts, {n_iter} iterations")

    for run in range(n_runs):
        print(f"\n{'='*60}")
        print(f"Run {run + 1}/{n_runs}")
        print(f"{'='*60}")

        optimizer = SimpleDiffusionContextualMOBO(
            objective_func=obj_func,
            model_type=model_type,
            problem_name=problem_name,
            acquisition_type=acquisition_type,
            diffusion_training_frequency=genai_training_frequency,
            top_p=genai_top_p,
            ddpm_num_candidates=genai_num_candidates,
            diffusion_min_data_points=8,
            diffusion_timesteps=100,
            diffusion_epochs=100,
            diffusion_batch_size=128,
            diffusion_hidden_dim=64,
            diffusion_num_layers=3,
            use_noise=False,
            scalar_type='HV',
            use_global_reference=True,
            use_batch_norm=use_batch_norm
        )

        n_initial_points = m_samples
        X_init = torch.zeros(n_initial_points * n_contexts, obj_func.input_dim + obj_func.context_dim)
        for i in range(n_contexts):
            start_idx = i * n_initial_points
            end_idx = (i + 1) * n_initial_points
            init_file = 'data/init_points_context_{}_{}_{}.pth'.format(i, obj_func.input_dim, n_initial_points)
            if os.path.exists(init_file):
                init_points = torch.load(init_file)
            else:
                sampler = qmc.LatinHypercube(d=obj_func.input_dim)
                init_points = torch.tensor(sampler.random(n=n_initial_points), dtype=torch.float32)
                os.makedirs('data', exist_ok=True)
                torch.save(init_points, init_file)
            X_init[start_idx:end_idx, :obj_func.input_dim] = init_points
            X_init[start_idx:end_idx, obj_func.input_dim:] = contexts[i].repeat(n_initial_points, 1)

        Y_init = obj_func.evaluate(X_init)
        X_opt, Y_opt = optimizer.optimize(X_init, Y_init, contexts, n_iter=n_iter,
                                          beta=temp_beta, run=run)

        run_data = {}
        print(f"\nFinal Results for Run {run + 1}:")
        for i, context in enumerate(contexts):
            context_key = tuple(context.numpy())
            if context_key in optimizer.context_pareto_fronts:
                run_data[context_key] = {
                    'pareto_set_history': optimizer.context_pareto_sets[context_key],
                    'pareto_front_history': optimizer.context_pareto_fronts[context_key],
                    'hv_history': optimizer.context_hv[context_key]
                }
                print(f"Context {i}: HV = {optimizer.context_hv[context_key][-1]:.4f}")
            else:
                print(f"Context {i}: No Pareto front found")

        method_name = f"BN_DDPM-CMOBO" if use_batch_norm else "DDPM-CMOBO"
        save_path = f'result/{problem_name}/{method_name}_{genai_training_frequency}_{genai_top_p}_{timestamp}_run_{run}.pth'
        torch.save(run_data, save_path)
        print(f"Run {run + 1} saved to {save_path}")

        fig, axes = plt.subplots(4, 4, figsize=(20, 20))
        fig.suptitle(f'DDPM-CMOBO: HV History (Run {run + 1})', fontsize=16)
        for i, context in enumerate(contexts):
            row, col = i // 4, i % 4
            ax = axes[row, col]
            context_key = tuple(context.numpy())
            if context_key in run_data:
                ax.plot(range(len(run_data[context_key]['hv_history'])),
                        run_data[context_key]['hv_history'])
            ax.set_title(f'Context {i}')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Hypervolume')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f'result/{problem_name}/{method_name}_{genai_training_frequency}_{genai_top_p}_{timestamp}_run_{run}.png')
        plt.close()
        print(f"Run {run + 1} completed!")

    print(f"\nAll {n_runs} runs completed. Results in: result/{problem_name}/")


def main():
    args = parse_arguments()
    print(f"Running {args.method_name} on {args.problem}")
    print(f"  Objectives: {args.n_objectives}, Variables: {args.n_variables}")
    print(f"  Tasks: {args.m_tasks}, Samples/task: {args.m_samples}")
    print(f"  Iterations: {args.n_iter}, Runs: {args.n_runs}")
    print(f"  Acquisition: {args.acquisition_type}, Model: {args.model_type}")

    if args.method_name == 'VAE-CMOBO':
        vae_optimization_loop_test(
            problem_name=args.problem,
            n_runs=args.n_runs,
            n_iter=args.n_iter,
            n_objectives=args.n_objectives,
            n_variables=args.n_variables,
            temp_beta=args.beta,
            model_type=args.model_type,
            m_tasks=args.m_tasks,
            m_samples=args.m_samples,
            acquisition_type=args.acquisition_type,
            genai_training_frequency=args.genai_training_frequency if args.genai_training_frequency is not None else 2,
            genai_top_p=args.genai_top_p if args.genai_top_p is not None else 0.1,
            genai_num_candidates=args.genai_num_candidates if args.genai_num_candidates is not None else 30000,
        )

    elif args.method_name == 'DDPM-CMOBO':
        simple_ddpm_optimization_loop_test(
            problem_name=args.problem,
            n_runs=args.n_runs,
            n_iter=args.n_iter,
            n_objectives=args.n_objectives,
            n_variables=args.n_variables,
            temp_beta=args.beta,
            model_type=args.model_type,
            m_tasks=args.m_tasks,
            m_samples=args.m_samples,
            acquisition_type=args.acquisition_type,
            genai_training_frequency=args.genai_training_frequency if args.genai_training_frequency is not None else 1,
            genai_top_p=args.genai_top_p if args.genai_top_p is not None else 0.1,
            genai_num_candidates=args.genai_num_candidates if args.genai_num_candidates is not None else 15000,
        )

    else:
        print(f"Unknown method: {args.method_name}. Choose VAE-CMOBO or DDPM-CMOBO.")


if __name__ == '__main__':
    torch.manual_seed(42)
    main()
