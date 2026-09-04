import { expect, test, type Page } from '@playwright/test';

const QWEN_MODEL = 'Qwen3.6-35B-A3B-Q6_K';

function serverProps(model: string) {
	return {
		chat_template: '{% if enable_thinking %}<think>{% endif %}',
		default_generation_settings: {
			n_ctx: 262144,
			params: { min_p: 0, temperature: 0.6, top_k: 20, top_p: 0.95 }
		},
		modalities: { audio: false, video: true, vision: true },
		model_alias: model,
		model_path: `/models/${model}.gguf`,
		total_slots: 1
	};
}

async function mockCommonEndpoints(page: Page, currentModel: () => string) {
	await page.route(/\/props(?:\?.*)?$/, (route) =>
		route.fulfill({ json: serverProps(currentModel()) })
	);
	await page.route('**/v1/models', (route) =>
		route.fulfill({
			json: {
				data: [{ id: currentModel(), object: 'model', owned_by: 'llamacpp' }],
				object: 'list'
			}
		})
	);
	await page.route('**/slots', (route) => route.fulfill({ json: [] }));
	await page.route('**/tools', (route) => route.fulfill({ json: [] }));
}

test('single-model selector exposes reasoning levels and sends the selected budget', async ({
	page
}) => {
	let requestBody: Record<string, unknown> | null = null;

	await mockCommonEndpoints(page, () => QWEN_MODEL);
	await page.route(/\/v1\/chat\/completions$/, async (route) => {
		requestBody = route.request().postDataJSON() as Record<string, unknown>;
		await route.fulfill({
			body: [
				'data: {"id":"chatcmpl-test","choices":[{"index":0,"delta":{"role":"assistant","content":"ok"},"finish_reason":null}]}',
				'data: {"id":"chatcmpl-test","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}',
				'data: [DONE]',
				''
			].join('\n\n'),
			contentType: 'text/event-stream',
			status: 200
		});
	});

	await page.goto('/');
	await page
		.getByRole('button', { name: /Qwen3\.6/i })
		.last()
		.click();
	await page.getByRole('menuitem', { name: /Reasoning/ }).hover();

	await expect(page.getByRole('menuitem', { name: /^Off$/ })).toBeVisible();
	await expect(page.getByRole('menuitem', { name: /^Low/ })).toContainText('512');
	await expect(page.getByRole('menuitem', { name: /^Medium/ })).toContainText('2,048');
	await expect(page.getByRole('menuitem', { name: /^High/ })).toContainText('8,192');
	await expect(page.getByRole('menuitem', { name: /^Max/ })).toContainText('Unlimited');

	await page.getByRole('menuitem', { name: /^High/ }).click();
	await page.locator('textarea').fill('teste');
	await page.locator('textarea').press('Enter');

	await expect.poll(() => requestBody).not.toBeNull();
	expect(requestBody).toMatchObject({
		chat_template_kwargs: { enable_thinking: true },
		reasoning_control: true,
		reasoning_format: 'auto',
		thinking_budget_tokens: 8192
	});
});

test('visible tab refreshes the displayed model after llama-server is replaced', async ({
	page
}) => {
	let currentModel = QWEN_MODEL;

	await mockCommonEndpoints(page, () => currentModel);
	await page.goto('/');
	await expect(page.getByRole('button', { name: /Qwen3\.6/i }).last()).toBeVisible();

	currentModel = 'Ornith-1.5-35B-Q6_K';

	await expect(page.getByRole('button', { name: /Ornith-1\.5/i }).last()).toBeVisible({
		timeout: 7000
	});
});
