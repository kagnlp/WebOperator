"""
Website Configuration for Dataset Organization

Add your website patterns here. Each website should have:
- patterns: List of regex patterns to match URLs
- description: Human-readable description

Example patterns:
- Exact URL: r'http://127\.0\.0\.1:3000'
- Domain: r'https://example\.com'
- Port range: r'http://127\.0\.0\.1:300[0-9]'
- Subdomain: r'https://[^.]+\.example\.com'
"""

WEBSITE_CONFIGS = {
    'map': {
        'patterns': [
            r'http://127\.0\.0\.1:3000',
            # r'https://127\.0\.0\.1:3000',
            # r'http://localhost:3000',
            # r'https://localhost:3000'
        ],
        'description': 'Map application (port 3000)'
    },

    # TODO: Add your 4 additional websites here
    # Template:
    # 'website_name': {
    #     'patterns': [
    #         r'http://127\.0\.0\.1:XXXX',  # Replace XXXX with port
    #         r'https://127\.0\.0\.1:XXXX',
    #         r'http://localhost:XXXX',
    #         r'https://localhost:XXXX'
    #     ],
    #     'description': 'Description of website'
    # },

    'shopping': {
        'patterns': [
            r'http://127\.0\.0\.1:7770',
            # r'https://127\.0\.0\.1:7770',
            # r'http://localhost:7770',
            # r'https://localhost:7770',
            # Add any shopping-related domain patterns
        ],
        'description': 'Shopping website (port 7770)'
    },

    'reddit': {
        'patterns': [
            r'http://127\.0\.0\.1:9999',
            # r'https://127\.0\.0\.1:9999',
            # r'http://localhost:9999',
            # r'https://localhost:9999',
            # Add any forum-related domain patterns
        ],
        'description': 'Forum website (port 9999)'
    },

    'shopping_admin': {
        'patterns': [
            r'http://127\.0\.0\.1:7780',
            # r'https://127\.0\.0\.1:7780',
            # r'http://localhost:7780',
            # r'https://localhost:7780',
            # Add any admin-related domain patterns
        ],
        'description': 'Admin dashboard (port 7780)'
    },

    'gitlab': {
        'patterns': [
            r'http://127\.0\.0\.1:8023',
            # r'https://127\.0\.0\.1:8023',
            # r'http://localhost:8023',
            # r'https://localhost:8023',
        ],
        'description': 'Gitlab website (port 8023)'
    }
}
